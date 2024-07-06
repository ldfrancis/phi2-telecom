


def create_queries(questions):
    return [

    "Represent this query for retrieving relevant documents: "+ question
    for question in questions
    ]

def create_keys(texts):
    return ["Represent this document for retrieval: " + text
        for text in texts
    ]