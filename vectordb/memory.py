"""
This module provides the Memory class that represents a memory storage system
for text and associated metadata, with functionality for saving, searching, and
managing memory entries.
"""
# pylint: disable = line-too-long, trailing-whitespace, trailing-newlines, line-too-long, missing-module-docstring, import-error, too-few-public-methods, too-many-instance-attributes, too-many-locals

from typing import List, Dict, Any, Union, Optional
import itertools
import numpy as np

from .chunking import Chunker
from .embedding import BaseEmbedder, Embedder
from .vector_search import VectorSearch
from .storage import Storage

from utils import Reader

# Import HNSW
HNSW_LOADED = True
try:
    from hnsw import HNSW, Node
except ImportError:
    print(
        "Warning: hnsw could not be imported. Install with 'pip install hnsw-lite'. "
        "Falling back to default search."
    )
    HNSW_LOADED = False

class Memory:
    """
    Memory class represents a memory storage system for text and associated metadata.
    It provides functionality for saving, searching, and managing memory entries.
    """

    def __init__(
        self,
        memory_file: str = None,
        chunking_strategy: dict = None,
        embeddings: Union[BaseEmbedder, str] = "normal",
        load_vec_files: bool = False,
        hnsw_preference: bool = False,
        hnsw_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the Memory class.

        :param memory_file: a string containing the path to the memory file. (default: None)
        :param chunking_strategy: a dictionary containing the chunking mode (default: {"mode": "sliding_window"}).
        :param embeddings: a string containing the name of the pre-trained model or an Embedder instance.
        :param load_vec_files: whether to load vector files (default: False).
        :param hnsw_preference: whether to use HNSW for search (default: False)
        :param hnsw_params: parameters for HNSW algorithm if used (default: None)
        """
        self.memory_file = memory_file

        if memory_file is None:
            self.memory = []
            self.metadata_memory = []
        else:
            load = Storage(memory_file).load_from_disk()
            self.memory = [] if len(load) != 1 else load[0]["memory"]
            self.metadata_memory = [] if len(load) != 1 else load[0]["metadata"]

        if chunking_strategy is None:
            chunking_strategy = {"mode": "sliding_window"}
        self.chunker = Chunker(chunking_strategy)

        self.metadata_index_counter = 0
        self.text_index_counter = 0

        if isinstance(embeddings, str):
            self.embedder = Embedder(embeddings)
        elif isinstance(embeddings, BaseEmbedder):
            self.embedder = embeddings
        else:
            raise TypeError("Embeddings must be an Embedder instance or string")

        # Initialize vector search for non-HNSW fallback
        self.vector_search = VectorSearch()
        
        # HNSW setup
        self.hnsw_preference = hnsw_preference and HNSW_LOADED
        self.hnsw_obj = None
        
        # Default HNSW parameters
        self.hnsw_params = {
            "M": 16,                 # Number of bi-directional links
            "ef_construction": 200,  # Size of dynamic candidate list for construction
            "space": "cosine"        # Distance metric: "cosine" or "l2"
        }
        
        # Update with user-provided parameters if any
        if hnsw_params:
            self.hnsw_params.update(hnsw_params)
            
        # Initialize HNSW if preferred
        if self.hnsw_preference:
            self._initialize_hnsw()
            
            # If we already have vectors, add them to the HNSW index
            if self.memory:
                for i, entry in enumerate(self.memory):
                    self.hnsw_obj.insert(
                        entry["embedding"],
                        metadata={"id": i, "text_index": entry["text_index"], "metadata_index": entry["metadata_index"]}
                    )
    
    def _initialize_hnsw(self):
        """Initialize HNSW index with current parameters."""
        print("Initializing HNSW with parameters:", self.hnsw_params)
        if not HNSW_LOADED:
            print("HNSW not available, falling back to default search.")
            self.hnsw_preference = False
            return
            
        self.hnsw_obj = HNSW(
            M=self.hnsw_params["M"],
            ef_construction=self.hnsw_params["ef_construction"],
            space=self.hnsw_params["space"]
        )

    def save(
        self,
        texts,
        metadata: Union[List, List[dict], None] = None,
        memory_file: str = None,
    ):
        """
        Saves the given texts and metadata to memory.

        :param texts: a string or a list of strings containing the texts to be saved.
        :param metadata: a dictionary or a list of dictionaries containing the metadata associated with the texts.
        :param memory_file: a string containing the path to the memory file. (default: None)
        """

        if not isinstance(texts, list):
            texts = [texts]

        if metadata is None:
            metadata = []
        elif not isinstance(metadata, list):
            metadata = [metadata]

        # Extend metadata to be the same length as texts, if it's shorter.
        metadata += [{}] * (len(texts) - len(metadata))

        for meta in metadata:
            self.metadata_memory.append(meta)

        meta_index_start = (
            self.metadata_index_counter
        )  # Starting index for this save operation
        self.metadata_index_counter += len(
            metadata
        )  # Update the counter for future save operations

        if memory_file is None:
            memory_file = self.memory_file

        text_chunks = [self.chunker(text) for text in texts]
        chunks_size = [len(chunks) for chunks in text_chunks]

        flatten_chunks = list(itertools.chain.from_iterable(text_chunks))

        embeddings = self.embedder.embed_text(flatten_chunks)

        text_index_start = (
            self.text_index_counter
        )  # Starting index for this save operation
        self.text_index_counter += len(texts)

        # accumulated size is end_index of each chunk
        for size, end_index, chunks, meta_index, text_index in zip(
            chunks_size,
            itertools.accumulate(chunks_size),
            text_chunks,
            range(meta_index_start, self.metadata_index_counter),
            range(text_index_start, self.text_index_counter),
        ):
            start_index = end_index - size
            chunks_embedding = embeddings[start_index:end_index]

            for chunk, embedding in zip(chunks, chunks_embedding):
                entry = {
                    "chunk": chunk,
                    "embedding": embedding,
                    "metadata_index": meta_index,
                    "text_index": text_index,
                }
                # Add to internal memory
                self.memory.append(entry)
                
                # If using HNSW, add to the HNSW index
                if self.hnsw_preference and self.hnsw_obj:
                    idx = len(self.memory) - 1
                    self.hnsw_obj.insert(
                        embedding, 
                        metadata={"id": idx, "text_index": text_index, "metadata_index": meta_index}
                    )

        if memory_file is not None:
            Storage(memory_file).save_to_disk([{"memory": self.memory, "metadata" :self.metadata_memory}])

    def search(
        self, query: str, top_n: int = 5, unique: bool = False, batch_results: str = "flatten", 
        use_hnsw: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for the most similar chunks to the given query in memory.

        :param query: a string containing the query text.
        :param top_n: the number of most similar chunks to return. (default: 5)
        :param unique: chunks are filtered out to unique texts (default: False)
        :param batch_results: if input is list of queries, results can use "flatten" or "diverse" algorithm
        :param use_hnsw: override the default HNSW preference (default: None)
        :return: a list of dictionaries containing the top_n most similar chunks and their associated metadata.
        """
        # Determine whether to use HNSW
        use_hnsw_for_search = self.hnsw_preference if use_hnsw is None else use_hnsw
        
        if isinstance(query, list):
            query_embedding = self.embedder.embed_text(query)
        else:
            query_embedding = self.embedder.embed_text([query])[0]

        # Return empty results if memory is empty
        if len(self.memory) == 0:
            return []

        # Use HNSW for search if preferred and available
        if use_hnsw_for_search and self.hnsw_obj:
            indices = self._search_with_hnsw(query_embedding, top_n)
        else:
            # Fallback to vector_search
            embeddings = [entry["embedding"] for entry in self.memory]
            indices = self.vector_search.search_vectors(
                query_embedding, embeddings, top_n, batch_results
            )
        
        if unique:
            unique_indices = []
            seen_text_indices = set()
            for i in indices:
                text_index = self.memory[i[0]]["text_index"]
                if text_index not in seen_text_indices:
                    unique_indices.append(i)
                    seen_text_indices.add(text_index)
            indices = unique_indices

        results = [
            {
                "chunk": self.memory[i[0]]["chunk"],
                "metadata": self.metadata_memory[self.memory[i[0]]["metadata_index"]],
                "distance": i[1],
            }
            for i in indices
        ]

        return results
    
    def _search_with_hnsw(self, query_embedding, top_n):
        """
        Perform search using HNSW.
        
        :param query_embedding: The embedding vector to search for
        :param top_n: Number of results to return
        :return: List of (index, distance) tuples
        """
        # Make sure HNSW is initialized
        if not self.hnsw_obj:
            if self.hnsw_preference:
                self._initialize_hnsw()
                # Add all vectors to the index if needed
                for i, entry in enumerate(self.memory):
                    self.hnsw_obj.insert(
                        entry["embedding"],
                        metadata={"id": i, "text_index": entry["text_index"], "metadata_index": entry["metadata_index"]}
                    )
            else:
                # Fall back to vector_search if HNSW is not preferred
                embeddings = [entry["embedding"] for entry in self.memory]
                return self.vector_search.search_vectors(query_embedding, embeddings, top_n)
        
        # Ensure query_embedding is a numpy array
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
            
        # Create a node for the query
        query_node = Node(query_embedding, -1)
        
        # Search using HNSW
        results = self.hnsw_obj.knn_search(query_node, top_n)
        
        # Convert HNSW results to the same format as vector_search results
        return [(res[1].metadata["id"], res[0]) for res in results]

    def clear(self):
        """
        Clears the memory.
        """
        self.memory = []
        self.metadata_memory = []
        self.metadata_index_counter = 0
        self.text_index_counter = 0
        
        # Reset HNSW index as well
        if self.hnsw_preference:
            self._initialize_hnsw()

        if self.memory_file is not None:
            Storage(self.memory_file).save_to_disk([{"memory": self.memory, "metadata" :self.metadata_memory}])

    def dump(self):
        """
        Prints the contents of the memory.
        """
        for entry in self.memory:
            print("Chunk:", entry["chunk"])
            print("Embedding Length:", len(entry["embedding"]))
            print("Metadata:", self.metadata_memory[entry["metadata_index"]])
            print("-" * 40)

        print("Total entries: ", len(self.memory))
        print("Total metadata: ", len(self.metadata_memory))

    def load_vector_files(self, vec_file: str, number_of_vectors: int = None):
        """
        Load vectors from .fvecs, .bvecs, or .ivecs files.

        :param vec_files: a path to the .fvecs, .bvecs, or .ivecs file.
        """ 
        reader = Reader(vec_file)
        data = reader.data
        print(f"Loading {len(data)} vectors from {vec_file}")

        for idx, data_vector in enumerate(data):
            self.metadata_memory.append({})
            meta_index_start = self.metadata_index_counter
            self.metadata_index_counter += 1

            entry = {
                "embedding": data_vector,
                "chunk": f"vector {idx}",
                "metadata_index": meta_index_start,
                "text_index": self.text_index_counter
            }
            
            self.memory.append(entry)
            
            # If using HNSW, add to the index
            if self.hnsw_preference and self.hnsw_obj:
                vector_idx = len(self.memory) - 1
                self.hnsw_obj.insert(
                    data_vector, 
                    metadata={"id": vector_idx, "text_index": self.text_index_counter, "metadata_index": meta_index_start}
                )
                
            if len(self.memory) % 10000 == 0:
                print(f"Loaded {len(self.memory)} vectors")
            if number_of_vectors is not None and len(self.memory) >= number_of_vectors:
                break

    def search_vector(
        self, query_vector: List[float], top_k: int, use_hnsw: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar vectors to the given query vector.

        :param query_vector: a list of floats containing the query vector.
        :param top_k: an integer representing the number of most similar vectors to return.
        :param use_hnsw: whether to use HNSW for this search (default: None, uses instance preference)
        :return: a list of dictionaries containing the top_k most similar vectors and their associated metadata.
        """
        # Determine whether to use HNSW
        use_hnsw_for_search = self.hnsw_preference if use_hnsw is None else use_hnsw
        
        # Return empty results if memory is empty
        if len(self.memory) == 0:
            return []
            
        # Use HNSW for search if preferred and available
        if use_hnsw_for_search and HNSW_LOADED:
            indices = self._search_with_hnsw(query_vector, top_k)
        else:
            # Fallback to vector_search
            embeddings = [entry["embedding"] for entry in self.memory]
            indices = self.vector_search.search_vectors(query_vector, embeddings, top_k)
        
        results = [
            {
                "chunk": self.memory[i[0]]["chunk"],
                "metadata": self.metadata_memory[self.memory[i[0]]["metadata_index"]],
                "distance": i[1],
            }
            for i in indices
        ]
        return results
        
    def set_hnsw_preference(self, prefer_hnsw: bool, params: Optional[Dict[str, Any]] = None):
        """
        Set whether to use HNSW for search and update parameters.
        
        :param prefer_hnsw: Whether to use HNSW for search
        :param params: Optional parameters for HNSW
        """
        # Update preference
        self.hnsw_preference = prefer_hnsw and HNSW_LOADED
        
        # Update parameters if provided
        if params:
            self.hnsw_params.update(params)
        
        # Reinitialize HNSW if needed
        if self.hnsw_preference:
            self._initialize_hnsw()
            
            # Add existing vectors to the index
            if self.memory:
                for i, entry in enumerate(self.memory):
                    self.hnsw_obj.insert(
                        entry["embedding"],
                        metadata={"id": i, "text_index": entry["text_index"], "metadata_index": entry["metadata_index"]}
                    )

