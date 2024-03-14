from lunary import LunaryCallbackHandler
from langchain_openai import ChatOpenAI

handler1 = LunaryCallbackHandler(app_id="07ff18c9-f052-4260-9e89-ea93fe9ba8c5")
handler2 = LunaryCallbackHandler(app_id="c70d70af-fc48-46a9-a2e9-d27aef85d20c")

chat = ChatOpenAI(
  callbacks=[handler1, handler2],
)
chat.invoke("Write a random string of 4 letters")