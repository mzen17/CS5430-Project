# CS5430 Project
<img width="2208" height="428" alt="image" src="https://github.com/user-attachments/assets/29b90632-a5de-4337-abae-b5169b52eb9b" />

## Running the code
To run, please run in the specific manner:
```uv run python main.py```

To run an individual component:

```uv run python vae.py```

```uv run python lem.py``` (depends on above)

```uv run python latent.py``` (depends on above)

```uv run python lla.py``` (depends on lem.py run)

Note: The paper was ran with epochs at 10000 epochs for VAE, but is set to 1000 default. The iterations of langevin is also set to 150. These can easily be modified in the vae.py/lem.py for full paper replication if have access to a GPU.