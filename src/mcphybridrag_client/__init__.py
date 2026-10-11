import asyncio


def main() -> None:
    from mcphybridrag_client.client import main as client_main

    asyncio.run(client_main())
