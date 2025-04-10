from yuki import Yuki

yuki = Yuki("Yuki", "Elias")

while True:
    inp = input(": ").strip()
    if inp.lower() == "clear_mem":
        yuki.vdb_client.reset()
    if inp.lower() == "exit":
        yuki.update_episodic()
        break

    response = yuki.msg(inp)
    print(response)
