

PHI2_MODEL_ID = "microsoft/phi-2"
EMBED_MODEL_ID = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"#"BAAI/llm-embedder"
EMBED_MODEL_TYPE = "SentenceTransformer"#"HuggingFace"

DATA_DIR = "data/"
SAMPLE_SUB_FILE = DATA_DIR + "SampleSubmission.csv"
TRAINING_FILE = DATA_DIR + "TeleQnA_training.txt"
TRAINING_ANSWER_FILE = DATA_DIR + "Q_A_ID_training.csv"
TESTING_FILE = DATA_DIR + "TeleQnA_testing1.txt"

EMBEDS_FILE = "data/doc/chunks/embeds.npy"
CHUNKS_FILE = "data/doc/chunks/chunks.npy"
SOURCES_FILE = "data/doc/chunks/sources.npy"

