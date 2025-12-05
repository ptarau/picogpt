from picogpt import PicoGPT

picoGPT = PicoGPT()
picoGPT.train_with_file("data/corpus.txt")
picoGPT.save_model("data/corpus.pt")
answer = picoGPT.ask("RL aligns the model")
print(answer)
