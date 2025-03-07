import argparse
import json
import logging
import colorlog
import os
import random
import requests
import asyncio
import sys
import time
import toml
import dataclasses
import subprocess
import secrets
import hashlib
import re
import glob
import datetime

from croniter import croniter
from dataclasses import dataclass
from mattermostautodriver import AsyncDriver
from typing import List, Optional
from urllib.parse import urlparse
from dacite import from_dict
from shutil import which
from randomgen import PCG32, ExtendedGenerator
from pytz import timezone

UNICODE_NUMBER_EMOJIS = [
	"one",
	"two",
	"three",
	"four",
	"five",
	"six",
	"seven",
	"eight",
	"nine",
	"keycap_ten"
]

# list of paths that are commonly used storing the kernel image
COMMON_KERNEL_LOCATIONS = [
	"/boot/vmlinuz",
	"/boot/vmlinuz-*",
	"/boot/vmlinux",
	"/boot/vmlinux-*",
	"/boot/kernel",
	"/boot/kernel-*",
	"/boot/kernel-image",
	"/boot/kernel-image-*",
	"/boot/kernel.img",
	"/boot/kernel.img-*",
	"/boot/kernel.efi",
	"/boot/kernel.efi-*",
	"/boot/kernel64",
	"/boot/kernel64-*",
	"/boot/kernel64.efi",
	"/boot/kernel64.efi-*",
	"/boot/kernel64.img",
	"/boot/kernel64.img-*",
]

@dataclass
class GeneralConfig:
	timezone: str
	random_org_api_key: Optional[str]

@dataclass
class LocaleConfig:
	vote_datetime_format: str
	vote_start: str
	vote_location_added: str
	vote_custom_location_disabled: str
	vote_custom_location_limit_reached: str
	vote_fortune: str
	vote_no_fortune: str
	vote_end: str
	vote_end_short: str
	vote_end_tie: str

@dataclass
class MattermostConfig:
	mattermost_url: str
	team_name: str
	access_token: str
	channel: str

	def create_mattermost_driver(self):
		# driver needs these as separate arguments
		url = urlparse(self.mattermost_url)

		# if no port is given, use the default port for the scheme
		port = url.port
		if port is None:
			port = 443 if url.scheme == 'https' else 80

		return AsyncDriver({
			'url': url.hostname,
			'token': self.access_token,
			'scheme': url.scheme,
			'port': port,
			'verify': True,
			'timeout': 30,
		})

@dataclass
class CustomLocationConfig:
	allow_custom_location: bool
	regex: List[str]

@dataclass
class PollConfig:
	options_count: int
	additional_reactions: List[str]
	locations: List[str]

@dataclass
class Poll:
	cron: str
	duration: int

@dataclass
class Config:
	general: GeneralConfig
	locale: LocaleConfig
	mattermost: MattermostConfig
	poll: PollConfig
	custom_location: CustomLocationConfig
	polls: List[Poll]


def seed_strong_random(config):
	# combine as many sources of entropy as possible to seed a cryptographically strong random number generator

	seed = 0
	sources = []

	# system random
	try:
		seed ^= secrets.randbelow(2**32)
		sources.append("system random")
	except Exception as e:
		logging.warning(f"Failed to get random number from system random: {e}")

	# random.org
	if config.general.random_org_api_key is not None:
		try:
			response = requests.post("https://api.random.org/json-rpc/4/invoke", json={
				"jsonrpc": "2.0",
				"method": "generateIntegers",
				"params": {
					"apiKey": config.general.random_org_api_key,
					"n": 1,
					"min": 0,
					# hard limit on random.org
					"max": 2000000000 - 1,
					"replacement": True
				},
				"id": 1
			})
			response.raise_for_status()
			logging.debug(f"Random.org response: {response.json()}")
			seed ^= response.json()["result"]["random"]["data"][0]
			sources.append("random.org")
		except Exception as e:
			logging.warning(f"Failed to get random number from random.org: {e}")

	# if on linux, hash the kernel image and add it to the seed
	if sys.platform == "linux":
		try:
			# find all kernel images
			kernel_images = []
			for location in COMMON_KERNEL_LOCATIONS:
				kernel_images.extend(glob.glob(location))
			if len(kernel_images) > 0:
				# hash the first kernel image
				with open(kernel_images[0], "rb") as f:
					seed ^= hashlib.sha256(f.read()).digest()
					sources.append("kernel")
		except Exception as e:
			logging.warning(f"Failed to hash kernel image: {e}")

	logging.info(f"Seeded random number generator with {seed} from sources: {', '.join(sources)}")

	# create secure random number generator with the seed
	rng = ExtendedGenerator(PCG32(seed=seed))

	# pull first number, since it's often spoiled and log it instead
	rnd = rng.random()
	logging.debug(f"First random number: {rnd}")

	return (rng, sources)

def get_fortune():
	try:
		fortune = None

		# use local fortune binary if available
		fortune_path = which('fortune')
		if fortune_path is not None:
			try:
				# run fortune and return the output
				logging.debug(f"Running fortune from {fortune_path}")
				fortune = subprocess.run([fortune_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
			except Exception as e:
				logging.warning(f"Failed to run fortune: {e}")

		if fortune is None:
			# use the fortune API as a fallback
			try:
				logging.debug("Getting fortune from yerkee.com")
				response = requests.get('http://yerkee.com/api/fortune')
				if response.status_code == 200:
					fortune = response.json()['fortune']
			except Exception as e:
				logging.warning(f"Failed to get fortune from yerkee.com: {e}")

		if fortune is None:
			try:
				logging.debug("Getting fortune from aphorismcookie")
				response = requests.get('https://aphorismcookie.herokuapp.com/')
				if response.status_code == 200:
					fortune = response.json()['data']['message']
			except Exception as e:
				logging.warning(f"Failed to get fortune from aphorismcookie: {e}")

		if fortune is not None:
			fortune = fortune.strip()

		return fortune

	except Exception as e:
		logging.error(f"Error getting fortune: {e}")
		return None

def load_config(config_path):
	logging.info(f"Loading config from {config_path}")
	with open(config_path, 'r', encoding='utf-8') as f:
		config_dict = toml.load(f)
		config = from_dict(data_class=Config, data=config_dict)

		# create a copy of the config, mask the access token and log the config object
		config_copy = dataclasses.replace(config,
			mattermost=dataclasses.replace(config.mattermost, access_token='***'),
			general=dataclasses.replace(config.general, random_org_api_key='***')
		)
		logging.info(f"Config: {config_copy}")

		return config

@dataclass
class PollOptions:
	locations: List[str]
	entropy_sources: List[str]

# stored state for resuming a poll after a restart or crash
@dataclass
class PollCheckpoint:
	post_id: str
	options: PollOptions
	close_time: datetime


@dataclass
class ConcludedPoll:
	# post id to refer to the poll
	post_id: str

	# options that were available in the poll (includes custom locations)
	options: PollOptions

	# list matching the options with the number of votes
	votes: List[int]

	# time when the poll was closed
	close_time: datetime

	# index of the winning location
	winner: int

# contains logic for running a poll and stores the state for serialization and resuming
@dataclass
class ActivePoll:
	bot: 'BotInstance'
	channel: dict
	checkpoint: PollCheckpoint

	# creates a new poll from the given options and duration
	@staticmethod
	async def new_poll(bot, close_time, poll_options):
		duration = close_time - datetime.datetime.now(datetime.UTC)

		logging.info(f"Creating new poll with options {poll_options} and duration {duration}")
		post_id = await ActivePoll.setup_poll(bot, poll_options, close_time)

		checkpoint = PollCheckpoint(post_id, poll_options, close_time)
		channel = await bot.get_channel()
		await bot.snapshot(checkpoint)
		return ActivePoll(bot, channel, checkpoint)

	@staticmethod
	async def resume(bot, checkpoint):
		duration = checkpoint.close_time - datetime.datetime.now(datetime.UTC)
		logging.info(f"Resuming poll with post ID {checkpoint.post_id} and options {checkpoint.options} and duration {duration}")

		# request message to ensure it exists
		channel = await bot.get_channel()
		logging.debug(f"Requesting poll message with ID {checkpoint.post_id}")
		post = await bot.driver.posts.get_post(checkpoint.post_id)
		if post is None:
			logging.error(f"Poll message with ID {checkpoint.post_id} not found, aborting poll")
			return

		return ActivePoll(bot, channel, checkpoint)

	async def send_message(self, data):
		channel = await self.bot.get_channel()
		post_data = {
			'channel_id': channel['id'],
			'root_id': self.checkpoint.post_id,
		}
		post_data.update(data)
		await self.bot.driver.posts.create_post(post_data)

	@staticmethod
	def format_message(poll_options, locale, close_time, timezone):
		# close time is utc, convert to local timezone
		close_time = close_time.astimezone(timezone)

		# format options as a numbered list
		options = "\n".join(f"{i + 1}. {option}" for i, option in enumerate(poll_options.locations))
		# format entropy sources
		entropy_sources = ", ".join(poll_options.entropy_sources)

		# remove whitespaces here instead of having user deal with it
		message = locale.vote_start.strip().format(
			locations=options,
			entropy_sources=entropy_sources,
			end_time=close_time.strftime(locale.vote_datetime_format)
		)
		return message.strip()

	async def add_location(self, location, author):
		locale = self.bot.config.locale
		locations = self.checkpoint.options.locations
		custom_location = self.bot.config.custom_location

		# check if custom locations are allowed, otherwise inform the user
		channel = await self.bot.get_channel()
		if not custom_location.allow_custom_location:
			# reply to the user that custom locations are not allowed
			logging.info(f"Custom locations are not allowed, informing {author}")
			await self.send_message({
				'message': locale.vote_custom_location_disabled
			})
			return
		# we add the reaction first, since the reaction limit might have been reached
		# never allow more options than we have reactions
		if len(locations) >= len(UNICODE_NUMBER_EMOJIS):
			logging.warning("Reaction limit reached, not adding location")
			await self.send_message({
				'message': locale.vote_custom_location_limit_reached
			})
			return

		# check if the location is already in the list
		if location in locations:
			logging.info(f"Location {location} already in list, not adding")
			return

		# add reaction to the poll message
		try:
			await self.bot.driver.reactions.save_reaction({
				'user_id': self.bot.driver.client.userid,
				'post_id': self.checkpoint.post_id,
				'emoji_name': UNICODE_NUMBER_EMOJIS[len(locations)]
			})
		except Exception as e:
			logging.error(f"Failed to add reaction to poll message: {e}")
			await self.send_message({
				'message': locale.vote_custom_location_limit_reached
			})
			return

		# add location to the list
		author = author.lstrip('@')
		location = f"{location} ({author})"
		locations.append(location)

		# reply to the user that the location was added
		await self.send_message({
			'message': locale.vote_location_added.format(location=location)
		})
		logging.info(f"Added location {location} to list")

		# update poll message with new location
		await self.update_poll_message()

		# update snapshot
		await self.bot.snapshot(self.checkpoint)

	async def update_poll_message(self):
		await self.bot.driver.posts.update_post(self.checkpoint.post_id, {
			'id': self.checkpoint.post_id,
			# we need to set this to true to keep the reactions
			'has_reactions': True,
			'message': ActivePoll.format_message(self.checkpoint.options, self.bot.config.locale, self.checkpoint.close_time, self.bot.timezone)
		})
		logging.debug(f"Updated poll message")

	async def handle_poll_message(self, author, message):		
		custom_location = self.bot.config.custom_location
		location = None
		# check if message passes the custom location regex
		for regex in custom_location.regex:
			# each regex must anchor itself, we do not enforce full match
			# each regex also exposes a "location" group that contains the location
			regex_match = re.search(regex, message, re.DOTALL)
			if regex_match is not None:
				location = regex_match.group("location")
				assert location is not None, f"Regex {regex} does not contain a 'location' group"
				break
		if location is None:
			return

		logging.info(f"Received custom location from {author}: {location}")
		await self.add_location(location.strip(), author)

	async def event_handler(self, event):
		# parse event json to dict
		try:
			event = json.loads(event)
		except json.JSONDecodeError:
			logging.error(f"Received event is not valid JSON: {event}")
			return
		logging.debug(f"Received event: {event}")

		# does not exist for status events
		ty = event.get("event")
		match ty:
			case "posted":
				# check if this is a reply to the poll message
				try:
					post = event["data"]["post"]
					try:
						# post data is json encoded again
						post = json.loads(post)
					except json.JSONDecodeError:
						logging.error(f"Post data is not valid JSON: {post}")
						return

					# ignore messages from the bot itself (careful when testing, you might ignore yourself)
					if self.bot.driver.client.userid == post["user_id"] and not self.bot.args.count_self:
						return

					if post["root_id"] == self.checkpoint.post_id:
						author = event["data"]["sender_name"]
						content = post["message"]
						logging.debug(f"Received reply to poll message ({self.checkpoint.post_id}) from {author}: {content}")
						await self.handle_poll_message(author, content)
				except KeyError:
					logging.warning(f"Received event does not match expected format: {event}")


	async def run(self):
		driver = self.bot.driver
		remaning_time = self.checkpoint.close_time - datetime.datetime.now(datetime.UTC)

		# spawn task for connecting and listening to websocket
		try:
			# driver does not expose actual websocket directly, so we can't control it
			websocket_future = driver.init_websocket(self.event_handler)
			websocket = driver.websocket
			assert websocket is not None, "unable to get websocket from driver, maybe 'API' changed"
			websocket_task = asyncio.create_task(websocket_future)

			# wait for the poll to end
			logging.info(f"Waiting for poll to end in {remaning_time.total_seconds()} seconds")
			await asyncio.sleep(remaning_time.total_seconds())
			websocket_task.cancel()
			websocket.disconnect()

			# close the poll
			await self.close_poll()
		finally:
			if websocket is not None:
				websocket.disconnect()

	async def close_poll(self):
		# fetch reactions from the poll message
		reactions = await self.bot.driver.reactions.get_reactions(self.checkpoint.post_id)
		logging.debug(f"Received reactions: {reactions}")
		
		# there is one element per reaction and per user, so we need to count them all up
		# initialize votes dict with 0 for each option index, to include options with zero votes in final message
		votes = {i: 0 for i in range(len(self.checkpoint.options.locations))}
		for reaction in reactions:
			emoji = reaction["emoji_name"]
			user = reaction["user_id"]
	
			# ignore reactions from the bot itself
			if user == self.bot.driver.client.userid and not self.bot.args.count_self:
				continue
			
			# only count valid count emojis
			if not emoji in UNICODE_NUMBER_EMOJIS:
				continue

			# add vote to the corresponding option and ensure that option index is valid
			index = UNICODE_NUMBER_EMOJIS.index(emoji)
			# only count votes for valid options
			if index < len(self.checkpoint.options.locations):
				votes[index] = votes.get(index, 0) + 1

		# sort votes by count
		votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
		logging.debug(f"Sorted votes: {votes}")

		# get all winners
		highest_vote = votes[0][1]
		winners = [i for i, v in votes if v == highest_vote]
		assert len(winners) > 0, "No winners found"
		logging.info(f"Winners: {winners}")

		# include stats in original message
		# for this we append the vote count to the location as "location (+votes)" ordered by votes
		locations_with_votes = []
		for pos, count in votes:
			locations_with_votes.append(f"{self.checkpoint.options.locations[pos]} (+{count})")

		start_message = ActivePoll.format_message(
			PollOptions(
				locations_with_votes,
				self.checkpoint.options.entropy_sources
			),
   		self.bot.config.locale,
			self.checkpoint.close_time,
			self.bot.timezone
		)
		# default to single winner
		winner = winners[0]
		end_message = self.bot.config.locale.vote_end.format(location=self.checkpoint.options.locations[winner])

		# and replace if there are multiple winners
		if len(winners) > 1:
			(rng, sources) = seed_strong_random(self.bot.config)
			winner = rng.random() * len(winners)
			winner = int(winner)
			
			# build list of all winning locations
			winning_locations = []
			for winner_item in winners:
				winning_locations.append(self.checkpoint.options.locations[winner_item])
			
			end_message = self.bot.config.locale.vote_end_tie.format(
				locations=", ".join(winning_locations),
				location=self.checkpoint.options.locations[winner],
				entropy_sources=", ".join(sources)
			)

		final_message = f"""
{start_message.strip()}

{end_message.strip()}
		""".strip()

		# construct short message after checking for tie, since winner can change inside the tie block
		short_end_message = self.bot.config.locale.vote_end_short.format(location=self.checkpoint.options.locations[winner])

		# post the final message
		await self.bot.driver.posts.update_post(self.checkpoint.post_id, {
			'id': self.checkpoint.post_id,
			'message': final_message,
			'has_reactions': True
		})

		# remote additional reactions, since they inflate actual participant count
		for reaction in self.bot.config.poll.additional_reactions:
			logging.debug(f"Removing reaction {reaction} from post {self.checkpoint.post_id}")
			await self.bot.driver.reactions.delete_reaction(self.bot.driver.client.userid, self.checkpoint.post_id, reaction)

		# post just the end message to the thread
		await self.send_message({
			'message': short_end_message
		})

		# record the result
		concluded_poll = ConcludedPoll(self.checkpoint.post_id, self.checkpoint.options, [v for i, v in votes], self.checkpoint.close_time, winner)
		await self.bot.record_result(concluded_poll)

	@staticmethod
	async def setup_poll(bot, poll_options, close_time):
		config = bot.config
		locale = config.locale

		# pull fortune at the start of the poll to hide the delay
		fortune = get_fortune()

		# post the poll message
		channel = await bot.get_channel()
		logging.debug(f"Posting poll message to channel {channel['name']} ({channel['id']})")
		post = await bot.driver.posts.create_post({
			'channel_id': channel['id'],
			'message': ActivePoll.format_message(poll_options, locale, close_time, bot.timezone)
		})
		post_id = post['id']

		# build reaction list
		reactions_to_add = []
		reactions_to_add.extend(config.poll.additional_reactions)
		for i in range(len(poll_options.locations)):
			reactions_to_add.append(UNICODE_NUMBER_EMOJIS[i])

		# post all reactions
		for reaction in reactions_to_add:
			logging.debug(f"Adding reaction {reaction} to post {post_id}")
			await bot.driver.reactions.save_reaction({
				'user_id': bot.driver.client.userid,
				'post_id': post_id,
				'emoji_name': reaction
			})

		# create a thread for the poll message
		thread_message = locale.vote_no_fortune
		if fortune is not None:
			thread_message = locale.vote_fortune.format(fortune=fortune)
		thread_message = thread_message.strip()

		logging.debug(f"Creating thread for poll message with ID {post_id}")
		thread = await bot.driver.posts.create_post({
			'channel_id': channel['id'],
			'root_id': post_id,
			'message': thread_message
		})

		logging.debug(f"Succesfully posted poll with ID {post_id} and thread ID {thread['id']}")
		return post_id


class BotInstance:
	def __init__(self, args):
		self.args = args
		self.config = load_config(args.config)
		self.driver = self.config.mattermost.create_mattermost_driver()

	async def get_channel(self):
		mattermost = self.config.mattermost
		channel = await self.driver.channels.get_channel_by_name_for_team_name(mattermost.team_name, mattermost.channel)
		return channel

	def select_option(self):
		# create strong random number generator
		(rng, sources) = seed_strong_random(self.config)

		# copy list and pull appropriate number of options
		options = self.config.poll.locations.copy()
		selection = []
		for i in range(self.config.poll.options_count):
			index = rng.random() * len(options)
			index = int(index)
			selection.append(options.pop(index))

		return (selection, sources)

	async def run_pre(self, after):
		# resolve timezone
		self.timezone = timezone(self.config.general.timezone)
		if self.timezone is None:
			logging.error(f"Invalid timezone: {self.config.general.timezone}")
			sys.exit(1)
		logging.info(f"Using timezone {self.timezone}, local time is {datetime.datetime.now(self.timezone)}")

		await self.driver.login()
		logging.info(f"Successfully logged in to Mattermost on instance {self.config.mattermost.mattermost_url}")

		channel = await self.get_channel()
		logging.info(f"Will post to channel {channel['name']} ({channel['id']})")

		# run action
		try:
			await after()
		except asyncio.CancelledError:
			logging.info("Received cancel signal, exiting")
			sys.exit(0)
		except KeyboardInterrupt:
			logging.info("Received keyboard interrupt, exiting")
			sys.exit(0)
		except Exception as e:
			logging.error(f"Error during action: {e}")
			sys.exit(1)

	async def run_once(self):
		close_time = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=self.args.duration)
		poll = await self.create_poll(close_time)
		await poll.run()

	async def create_poll(self, close_time):
		(selection, sources) = self.select_option()
		logging.info(f"Selected options: {selection} from sources: {', '.join(sources)}")
		poll = await ActivePoll.new_poll(self, close_time, PollOptions(selection, sources))
		return poll

	async def run_cron(self):
		# TODO: check if there are any active polls and resume them

		assert len(self.config.polls) > 0, "No polls configured"

		while True:
			# resolve in local timezone, since user input is in local time
			base_time = datetime.datetime.now(self.timezone)

			# check all polls to find the next one
			upcoming_polls = []
			for cron in self.config.polls:
				cron_time = croniter(cron.cron, base_time)
				next_time = cron_time.get_next(datetime.datetime)
				upcoming_polls.append((next_time, cron))
    
			assert len(upcoming_polls) > 0, "No upcoming polls found"

			# sort by time and select the next poll
			upcoming_polls.sort(key=lambda x: x[0])
			next_poll = upcoming_polls[0]
			wait_time = next_poll[0] - base_time
			logging.info(f"Next poll is at {next_poll[0]}, waiting {wait_time.total_seconds()} seconds")
			await asyncio.sleep(wait_time.total_seconds())

			# create the poll
			close_time = next_poll[0] + datetime.timedelta(seconds=next_poll[1].duration)
			poll = await self.create_poll(close_time)

			# run poll in background so we can queue multiple polls
			asyncio.create_task(poll.run())

			# wait a bit to avoid triggering the same cron job multiple times
			await asyncio.sleep(5)
			

	async def check_resume(self):
		# TODO: implement db lookup to check for active polls
		return [] # TODO: return a list of checkpoints

	async def snapshot(self, checkpoint):
		# TODO: implement database storage
		logging.warning(f"NOT IMPLEMENTED: Snapshotting checkpoint {checkpoint}")

	async def record_result(self, concluded_poll):
		# TODO: implement database storage
		logging.warning(f"NOT IMPLEMENTED: Recording result {concluded_poll}")

if __name__ == "__main__":
	# set up logger to info, so we can log during setup
	handler = colorlog.StreamHandler()
	handler.setFormatter(colorlog.ColoredFormatter("[%(log_color)s%(levelname)s%(reset)s] %(message)s"))
	logging.basicConfig(handlers=[handler], level=logging.INFO)

	parser = argparse.ArgumentParser(description='Lunch spot voting bot.')
	parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
	parser.add_argument('--log', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
	parser.add_argument('--count-self', action='store_true', help='Count the bot itself as a voter')
	subparsers = parser.add_subparsers(dest='command', help='Commands')
 
	once = subparsers.add_parser('once', help='Run the bot once and exit')
	once.add_argument('--duration', type=int, required=True, help='Duration of the poll in minutes')
	cron = subparsers.add_parser('cron', help='Run the bot in cron mode')

	args = parser.parse_args()

	# set log level according to command line argument
	log_level = getattr(logging, args.log.upper(), logging.INFO)
	logging.info(f"Switching to log level {args.log}")
	logging.getLogger().setLevel(log_level)

	
	bot = BotInstance(args)
	match args.command:
		case 'once':		
			after_action = bot.run_once
		case 'cron':
			after_action = bot.run_cron
		case _:
			logging.error("No command given")
			parser.print_help()
			sys.exit(1)
	
	asyncio.run(bot.run_pre(after_action))
