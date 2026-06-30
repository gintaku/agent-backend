from .retriever import retrieve


# Use the retrieve function to get information from the knowledge base
query = "additional checks that need to pass before the github Copilot responses are subbmitted to the user"
result = retrieve(query)
# write the result to a file for testing purposes
with open("test_output.txt", "w", encoding="utf-8") as f:
    f.write(result)