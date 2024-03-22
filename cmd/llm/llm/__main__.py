# Standard Libraries
import asyncio

# 3rd Party Libraries
from llm.app import App


def main():
    asyncio.run(App().start())


if __name__ == "__main__":
    main()
