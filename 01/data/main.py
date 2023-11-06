import discord
import os
from dotenv import load_dotenv
import json
import csv

load_dotenv()
token = os.getenv("DISCORD_TOKEN")
if token == None:
    exit("Could not find $DISCORD_TOKEN env var.")

intents = discord.Intents.default()
intents.guilds = True
intents.message_content = True
intents.members = True

client = discord.Client(intents=intents)

contributor_ids_json_path = "contributor_ids.json"
CSV_FILENAME = "training_data.csv"

if not os.path.exists(contributor_ids_json_path):
    initialized_from_file = False
    contributor_ids = {"tankbottoms": 921302904546140191}
else:
    initialized_from_file = True
    with open(contributor_ids_json_path, "r") as file:
        contributor_ids = json.load(file)

JB_GUILD_ID = 775859454780244028
CONTRIBUTOR_ROLE = 865459358434590740

message_buffer = []
BUFFER_SIZE = 1000

with open(CSV_FILENAME, "w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(['author_id', 'author_global_name', 'message_content'])

async def flush_messages_to_csv():
    with open(CSV_FILENAME, "a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(message_buffer)
    message_buffer.clear()

@client.event
async def on_ready():
    print(f"Successfully logged in as {client.user}")

    jb_guild = await client.fetch_guild(JB_GUILD_ID)
    if not initialized_from_file:
        async for member in jb_guild.fetch_members(limit=None):
            if (
                member.get_role(CONTRIBUTOR_ROLE) is not None
                and member.global_name is not None
            ):
                print(f"Found contributor. {member.global_name}: {member.id}")
                contributor_ids[member.global_name] = member.id
        with open(contributor_ids_json_path, "w") as file:
            json.dump(contributor_ids, file, indent=2)

    contributor_ids_set = set(contributor_ids.values()) # Use set for faster lookup

    channels = await jb_guild.fetch_channels()
    message_count = 0

    for channel in channels:
        if isinstance(channel, discord.TextChannel):
            try:
                async for message in channel.history(limit=None):
                    if message.author.id in contributor_ids_set:
                        message_count += 1
                        message_buffer.append([message.author.id, message.author.global_name, message.clean_content])

                        if message_count % BUFFER_SIZE == 0:
                            await flush_messages_to_csv()
                            print(f"Wrote {message_count} messages...")

                if message_buffer:
                    await flush_messages_to_csv()

            except discord.Forbidden:
                print(f"Do not have permissions to read message history in {channel.name}.")
            except discord.HTTPException as e:
                print(f"Request to read {channel.name} failed: {e}")

    await client.close()

client.run(token)
