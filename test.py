import argparse


# creating object for argparse
if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--name", "-n", default="anjan", type=str)
    args.add_argument("--age", "-a", default=30.0, type=float)
    parse_args=args.parse_args() # parse_arg is method name
    
    print(parse_args.name,parse_args.age)
