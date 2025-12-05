from picogpt import PicoGPT

picoGPT = PicoGPT()
picoGPT.load_model("pico.pt")
answer = picoGPT.ask("RL aligns the model")
print(answer)
