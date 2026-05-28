import random
from poke_env.player import Player

class RandomBot(Player):
    async def choose_move(self, battle):
        if battle.available_moves:
            return self.create_order(
                random.choice(battle.available_moves)
            )

        if battle.available_switches:
            return self.create_order(
                random.choice(battle.available_switches)
            )

        return self.choose_default_move()
