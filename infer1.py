from picogpt import PicoGPT

picoGPT = PicoGPT()
picoGPT.load_model("data/triples.pt")
answer = picoGPT.ask("at_risk_of_being_replaced_by")
print(answer)
while True:
    prompt = input(":--> ")
    if not prompt:
        break
    answer = picoGPT.ask(prompt)
    print(answer)
