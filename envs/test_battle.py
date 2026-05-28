import asyncio
import logging
from poke_env.player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


logging.basicConfig(level=logging.INFO)


async def main():
    bot = RandomPlayer(
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration(
            "usuhuu",
            "password"
        ),
        server_configuration=LocalhostServerConfiguration
    )

    print("Bot username:", bot.username)
    print("Trying to connect...")
    print("Waiting for challenge...")
    await bot.accept_challenges("usuhuu1", 1)
    print("Challenge accepted!")

asyncio.run(main())