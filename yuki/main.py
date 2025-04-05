from yuki import Yuki

yuki = Yuki()

while True:
    inp = input(": ").strip()
    if inp.lower() == "exit":
        break

    response = yuki.msg(inp)
    print(response)
