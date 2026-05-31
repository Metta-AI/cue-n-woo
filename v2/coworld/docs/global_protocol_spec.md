# Steering Game Global Protocol

Global viewers connect to `/global`. The server sends public snapshots of the episode state.

The global snapshot includes phase, countdown timer, public questions, aggregate counts, and final scores when the game
ends. It does not include the hidden steering concept while the game is active.

