# GET  /threads/{thread_id}/messages
# POST /threads/{thread_id}/messages
# {
#   "content": "Explain photosynthesis"
# }

# POST /threads/{id}/messages
#               │
#               ▼
#        create User Message
#               │
#               ▼
#          create Run
#               │
#               ▼
#        execute RAG pipeline
#               │
#               ▼
#     create Assistant Message
#               │
#               ▼
#           return result