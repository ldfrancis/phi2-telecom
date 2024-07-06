#! /usr/bin/env bash


./scripts/unpack_rel18.sh

if ! grep doc2txt .states &> /dev/null
then
echo "Converting docs to txt files ..."
python ./scripts/docs2txt.py
echo doc2txt >> .states
echo "Done converting docs to txt files"
else
echo "Converted docs to txt files"
fi


if ! grep ragchunks .states &> /dev/null
then
echo "Creating document chunks for RAG"
if python ./scripts/rag_chunks.py 
then
echo ragchunks >> .states
echo "Done creating document chunks for RAG"
else
exit 1
fi
else
echo "Created document chunks for RAG"
fi

if ! grep chunkembeddings .states &> /dev/null
then
echo "Creating chunk embeddings"
if python ./scripts/chunk_embeddings.py
then
echo chunkembeddings >> .states
echo "Done creating chunk embeddings"
else
exit 1
fi
else
echo "Created chunk embeddings" 
fi

