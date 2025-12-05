from picogpt import PicoGPT

picoGPT = PicoGPT()
picoGPT.train_with_file("data/triples.tsv")
picoGPT.save_model("data/triples.pt")
answer = picoGPT.ask("journalists	at risk to ")
print(answer)
