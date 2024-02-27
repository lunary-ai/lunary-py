import lunary
import os
from openai import OpenAI

client = OpenAI()

lunary.monitor(client)

def my_llm_agent(input):
  res = client.chat.completions.create(
    model="gpt-3.5-turbo", 
    messages=input 
  )
  return res.choices[0].message.content

dataset = lunary.get_dataset("test") # Replace with the name of the dataset you want to use

for item in dataset:
  result = my_llm_agent(item.input)
  print(result)

  passed, results = lunary.evaluate(
    checklist="pirate", # Replace with the name of the checklist you want to use
    input=item.input,
    output=result,
    ideal_output=item.ideal_output
  )

  if passed:
    print("Test passed!")
  else:
    print("Test failed!")

  print(results)
