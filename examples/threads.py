import lunary
from dotenv import load_dotenv
import time

load_dotenv()

thread = lunary.open_thread()

thread.track_message({
  "role": "user",
  "content": "Hello!"
}, "user123")

time.sleep(0.5)

thread.track_message({
  "role": "assistant",
  "content": "How can I help you?"
})

time.sleep(0.5)

thread.track_message({
  "role": "user",
  "content": "I have a question about your product."
}, "user123")
time.sleep(0.5)

msg_id = thread.track_message({
  "role": "assistant",
  "content": "Sure, happy to help. What's your question?"
})
#time.sleep(0.5)

# lunary.track_feedback(msg_id, {
#     "thumbs": "down",
#     "comment": "I don't feel comfortable sharing my credit card number."
# })

