from picogpt import PicoGPT

picoGPT = PicoGPT()
picoGPT.train_with_file("corpus.txt")
picoGPT.save_model("pico.pt")
answer = picoGPT.ask("RL aligns the model")
print(answer)
