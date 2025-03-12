# SERA - The Discord Restaurant Assistant

## Running the bot locally

1. Run the reservation server: `python reservation_server.py`
2. Run the bot: `python bot.py`

## Deploying the bot

The bot is hosted in a GCP VM instance.

To run the bot, navigate into the repository directory. There are two [tmux](https://github.com/tmux/tmux/wiki) sessions
running: `restaurant_server.py` and `bot.py`.
To enter one of these sessions, run:

```
tmux attach -t restaurant-server
```

or:

```
tmux attach -t bot-server
```

To detach from a tmux session, press `CTRL+B D`.

Make sure to run `python reservation_server.py` and `python bot.py` in their respective tmux sessions.

## Functionality

This bot is able to:

- Search for restaurants
- Make restaurant reservations
- Split the bill
- Have natural language conversations with users

## Todos

- [x] Artur: Debug passing images to the split bill
- [ ] Artur: Debug why his tool is dumb
- [ ] Sherry: Move truncation of tool output into the search restaurants tool
- [ ] Sherry: Give extra details for one specific restaurant
- [ ] Rishi: After the reservation is made, make the bot post the confirmation number and details to the channel
- [ ] Rishi: Prompt the user for a time range in case the specific time is not available
- [ ] Eric: Add tooling to save user's phone number, location, name, and preferences to a database
- [ ] Eric: Add tooling to retrieve all of a user's details from the database
