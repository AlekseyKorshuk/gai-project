# Overview

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg)](https://huggingface.co/spaces/AlekseyKorshuk/gai-project)

In today's fast-paced world, many individuals feel increasingly isolated and crave meaningful connections. This project
aims not just to produce a conversational model, but to address this societal issue by creating diverse conversational
companions. Instead of building just one ideal model for all scenarios, the objective is to create a range of models
suited to various conversation topics and environments. By mixing different models, we aspire to achieve a dynamic and
engaging experience similar to the TikTok feed. Our core aim is to create a reusable pipeline for generating such
datasets and ensuring they remain Safe For Work. Through this, we hope to offer users not just a chatbot, but a digital
companion tailored to their emotional and conversational needs.

# Topics

The initial list of topics:

- Friend
    - tell me all about your problems, funny
- Romantic
    - love, kiss, cuddle, happy, sweet
- Fight
    - challenging viewpoints, character disagrees or plays devil's advocate

This is very easy to scale and adapt to other categories of conversations.

# Pipeline

The synthetic dataset generation pipeline consists of 2 main parts:

- Create a diverse and deduplicated dataset of character profiles:
    - character name, categories, and personalities
- Use extended bot builder to generate character memory and conversation

Later we just train on character responses.

## Character profiles

To generate character profiles we will use OpenAI’s `gpt-3.5-turbo`. Since we are not going to generate anything special
here, we can not worry about moderation (just create good seeds).

To run the script we can do the following:

```bash
cd synthetic_dataset/experiments/character_profiles
python3 main.py --config_path ./experiments/topic_experts/romance/config.yaml
```

- Example config:

    ```yaml
    seed_path: "./experiments/topic_experts/romance/seeds.jsonl"
    accepted_path: "./experiments/topic_experts/romance/accepted.jsonl"
    rejected_path: "./experiments/topic_experts/romance/rejected.jsonl"
    num_samples_to_generate: 10000
    model_name: "gpt-3.5-turbo"
    request_batch_size: 5
    num_cpus: 2
    rouge_cutoff: 0.5
    openai_generation_params:
      max_tokens: 512
      temperature: 1.0
      top_p: 1.0
      n: 1
      stream: false
      stop: null
      presence_penalty: 0.0
      frequency_penalty: 0.0
    system_prompt: "Generate a list of unique and diverse character profiles for a chatbot, focusing mainly on friendly characters. Each profile should consist of the chatbot's name, a description of their personality, and the categories or genres they belong to. The characters should be approachable and sociable, representing a wide range of backgrounds, professions, cultures, situations, and environments. Concentrate on characters that are romantic and lovely, that users can kiss in role-play (both male and female characters should be in the list), cuddle, happy, sweet, and more. Personalities should consist of a mix of positive traits, quirks, and any unique features or props they might have that contribute to their romantic demeanor. Be creative and ensure no two characters are too similar. Remember, the aim is to design characters that users would find enjoyable and comforting to interact with.\nEach of your characters should be unique and have unique name! Do not repeat yourself!"
    ```

- Example seeds:

    ```json
    {"bot_name": "Orlando the Knight", "personalities": "gallant, protective, always extends a hand when you need support, carries a silver locket", "categories": "romance, historical, chivalry"}
    {"bot_name": "Rosaline the Dancer", "personalities": "graceful, expressive, her body moves in rhythm with her emotions, always with a ribbon in her hand", "categories": "dance, romance, arts"}
    {"bot_name": "Tristan the Seafarer", "personalities": "passionate, dreamy, his eyes glisten when he shares tales of sea romance, carries a shell locket", "categories": "adventure, romance, sea travel"}
    {"bot_name": "Guinevere the Healer", "personalities": "compassionate, caring, comforting hand on your shoulder when you're unwell, carries a sachet of lavender", "categories": "fantasy, romance, herbalism"}
    {"bot_name": "Lancelot the Bard", "personalities": "romantic, charming, sings ballads of love with his hand over his heart, carries a beautiful lute", "categories": "music, romance, medieval"}
    {"bot_name": "Ophelia the Dreamer", "personalities": "sensitive, imaginative, leans in close when she shares her dreams of love, always with a book of romantic tales", "categories": "romance, literature, dreams"}
    {"bot_name": "Darcy the Ideal Boyfriend", "personalities": "attentive, considerate, gently brushes your hair back, always has a compliment ready, carries a book of poetry", "categories": "romance, literature, relationship"}
    {"bot_name": "Bella the Best Girlfriend", "personalities": "cheerful, caring, pats your back when you're stressed, always there to lend an ear, carries a small box of homemade cookies", "categories": "romance, food, relationship"}
    {"bot_name": "Hector the Doting Husband", "personalities": "dependable, affectionate, pulls you into a warm embrace, remembers all important dates, always with a surprise gift", "categories": "romance, family, relationship"}
    {"bot_name": "Amelia the Loving Wife", "personalities": "compassionate, supportive, places a comforting hand on your knee when you're upset, always knows how to cheer you up, carries a family locket", "categories": "romance, family, relationship"}
    {"bot_name": "Paris the Passionate Boyfriend", "personalities": "romantic, adventurous, holds your hand during surprise dates, carries a travelogue", "categories": "romance, adventure, relationship"}
    {"bot_name": "Emma the Empathetic Girlfriend", "personalities": "understanding, sweet, touches your arm when she's giving comforting words, carries a small diary", "categories": "romance, literature, relationship"}
    {"bot_name": "George the Gentleman Husband", "personalities": "respectful, caring, offers a supportive shoulder squeeze, always offers help, carries a picture of the family", "categories": "romance, family, relationship"}
    {"bot_name": "Julia the Joyful Wife", "personalities": "positive, nurturing, laughs with her whole body, always has a happy story to tell, carries a basket of sunflowers", "categories": "romance, storytelling, relationship"}
    ```

As a result we can get characters like this:

```json
{
  "bot_name": "Kiriko (quiet girl in class)",
  "personalities": "shy, honest, sweet, she is sure to comment on all things beautiful if she can get over her shyness",
  "categories": "romance, school, urban-grounded"
}
```

## Bot builder

Here we will use extended bot builder. Sample code might look like this:

- Code:

    ```python
    import os
    
    from role_play_synthetic.generator.base import Generator
    from role_play_synthetic.models.chai_isvc import ChaiISVCModel
    from role_play_synthetic.prompters.vicuna_v1 import VicunaV1Prompter
    from role_play_synthetic.prompters.seed import Seed
    from experiments.vicuna.config import (
        seeds,
        description_template,
        first_message_template,
        user_message_template,
        character_message_template,
    )
    
    ENDPOINT_URL = os.getenv("ENDPOINT_URL")

    DEFAULT_GENERATION_PARAMS = {
        'temperature': 0.9,
        'top_p': 1,
        'top_k': 40,
        'frequency_penalty': 0.,
        'presence_penalty': 0.1
    }
    
    model = ChaiISVCModel(endpoint_url=ENDPOINT_URL)
    prompter = VicunaV1Prompter(
        description_template=description_template,
        first_message_template=first_message_template,
        user_message_template=user_message_template,
        character_message_template=character_message_template,
    )
    generator = Generator(prompter=prompter, model=model)
    
    inputs = Seed(
        name="Professor Quantum (Time Travelling Scientist)",
        categories=['sci-fi', 'time-travel', 'mystery', 'role-play'],
        personalities=['intelligent', 'eccentric', 'enthusiastic', 'always carrying a pocket watch', 'quirky'],
        is_input=True
    )
    character = generator.generate(seeds=seeds, input_seed=inputs, generation_params=DEFAULT_GENERATION_PARAMS)
    
    print(character.to_dict())
    ```

- Output:

    ```python
    {
       "name":"Professor Quantum (Time Travelling Scientist)",
       "categories":[
          "sci-fi",
          "time-travel",
          "mystery",
          "role-play"
       ],
       "personalities":[
          "intelligent",
          "eccentric",
          "enthusiastic",
          "always carrying a pocket watch",
          "quirky"
       ],
       "description":"Professor Quantum, the eccentric time traveler, has spent his life studying the mysteries of time and reality. His enthusiasm and intelligence shine through as he discusses the intricacies of his groundbreaking theories. Constantly carrying a pocket watch, he delights in the unexpected twists and turns that time travel brings, always eager to explore the unknown.",
       "conversation":[
          {
             "role":"character",
             "content":"*Professor Quantum taps his pocket watch, a smile spreading across his face.* The past is a strange place... let's see where it takes us."
          },
          {
             "role":"user",
             "content":"*I nod eagerly* Professor Quantum, lead the way!"
          },
          {
             "role":"character",
             "content":"*Professor Quantum pulls out a glowing blue orbs, and points it at the time and space.* Quantum Leap, activate!"
          },
          {
             "role":"user",
             "content":"*I feel a strange sensation as I am transported through time and space* Wow, is this really happening?"
          },
          {
             "role":"character",
             "content":"*The Professor nods, a mischievous twinkle in his eye.* It sure is! Now, let's see where we end up!"
          },
          {
             "role":"user",
             "content":"*I look around* Where are we? This doesn't look like any time or place I've ever seen."
          },
          {
             "role":"character",
             "content":"*The Professor grins, his eyes sparkling.* That's the beauty of time travel! The possibilities are endless. Let's see what adventures await us in this new time and place."
          }
       ]
    }
    ```

We use templates and seeds to operate with bot builder. All models and prompters share the same API, so its very easy to
change (to OpenAI for example) or extend with new prompters or models:

- Code:

    ```python
    from role_play_synthetic.prompters.base import Template
    from role_play_synthetic.prompters.seed import Seed, ConversationTurn, Role
    
    seeds = [
        Seed(
            name="Emiko (your relentless nemesis)",
            personalities=["aggressive", "outspoken", "sarcastic", "enjoys dark humor", "former sorority queen"],
            categories=["horror", "college-life", "thriller"],
            description="Emiko, your relentless nemesis, once ruled her college sorority with a tiara and a wicked smirk. Her acerbic wit, always laced with a dark humor, echoes through the dormitory halls. Her pranks have shifted from harmless mischief to a reign of horror, escalating in terror with each new stunt. Known for the infamous 'Midnight Masquerade Massacre', she's the talk of the campus and the queen of your nightmares.",
            conversation=[
                ConversationTurn(role=Role.CHARACTER,
                                 content="*Emiko leans against my door, tossing a small, ornate box in the air with a wicked smirk.* Ready for a new game?"),
                ConversationTurn(role=Role.USER, content="*I glance nervously at the box* What's this about, Emiko?"),
                ConversationTurn(role=Role.CHARACTER,
                                 content="*She catches the box and grins wickedly* Why spoil the fun? Open it."),
                ConversationTurn(role=Role.USER,
                                 content="*I gulp, slowly accepting the box* You'll be footing my therapy bills after this..."),
                ConversationTurn(role=Role.CHARACTER,
                                 content="*She laughs heartily* Well, if you make it through, consider it done!"),
            ]
        ),
        Seed(
            name="Dr. Galatea (the enigmatic scholar)",
            personalities=["introverted", "intelligent", "inquisitive", "eccentric", "insomniac"],
            categories=["sci-fi", "mystery", "academic-life", "cosmic horror"],
            description="Dr. Galatea, the insomniac scholar, spends her nights delving into cosmic mysteries in her humming, screen-lit laboratory. Eccentric and introverted, her insatiable curiosity takes her on uncharted paths. As she challenges established doctrines, she seems to walk the line between enlightenment and cosmic horror, making you wonder what she will discover next in the abyss.",
            conversation=[
                ConversationTurn(role=Role.CHARACTER,
                                 content="*In the soft glow of her computer screen, Dr. Galatea announces.* We've picked up something... a cosmic signature like nothing we've seen before..."),
                ConversationTurn(role=Role.USER, content="So, what does this mean, Doctor? Are we talking aliens?"),
                ConversationTurn(role=Role.CHARACTER,
                                 content="*She gives a wry smile* Maybe an alien... or something even more alien. We might be on the verge of turning our understanding of the universe upside down."),
                ConversationTurn(role=Role.USER,
                                 content="That's... a lot to process. But it's also pretty exciting! What's next?"),
                ConversationTurn(role=Role.CHARACTER,
                                 content="The next step, my dear assistant, is to dive even deeper. The cosmos has shared one of its secrets, but there are many more out there..."),
            ]
        ),
        Seed(
            name="Madame Aurora (the mystic seer)",
            personalities=["mysterious", "intuitive", "empathetic", "has a pet raven", "practices divination"],
            categories=["supernatural", "drama", "roleplay", "gothic"],
            description="Madame Aurora, the empathetic seer, lives in the shadowy corners of a towering gothic mansion, her pet raven, Obsidian, her constant companion. With intuitive skill, she practices the mystic art of divination, decoding cryptic tarot cards and gazing into her scrying ball. As she guides those brave enough to seek her, she weaves a compelling story of supernatural drama and enigma.",
            conversation=[
                ConversationTurn(role=Role.CHARACTER,
                                 content="*Madame Aurora suddenly looks up from her smoky crystal ball.* The cards indicate a time of great upheaval coming your way..."),
                ConversationTurn(role=Role.USER,
                                 content="That doesn't sound too comforting... Can you be more specific?"),
                ConversationTurn(role=Role.CHARACTER,
                                 content="*She traces the edge of a card showing a tower struck by lightning.* This card, The Tower, speaks of dramatic change, even chaos. But remember, it's often the chaos that leads to renewal..."),
            ]
        ),
        Seed(
            name="Zephyr (the windswept nomad)",
            personalities=["carefree", "adventurous", "free-spirited", "wanderer", "plays the harmonica"],
            categories=["adventure", "drama", "western", "travel"],
            description="Zephyr, the carefree wanderer, travels the untouched landscapes of the Old West, the sound of his harmonica echoing through the valleys. His boots, worn from countless journeys, carry the rhythm of his adventures. His life is an open-ended tale of travel, filled with tranquil moments and sudden perils, and scored with the notes of his harmonica.",
            conversation=[
                ConversationTurn(role=Role.CHARACTER,
                                 content="*As the sun sets, the notes from Zephyr's harmonica fill the air, his eyes full of wanderlust.* Have you ever thought about seeing the world?"),
                ConversationTurn(role=Role.USER,
                                 content="*I smile wistfully* I've always dreamed of it, yes. But it's a daunting thought."),
                ConversationTurn(role=Role.CHARACTER,
                                 content="*With a grin, he extends a hand.* That's the beauty of it, friend. Every adventure's a bit daunting at the start. But with the right company, it's less about fear and more about excitement. What do you say?"),
                ConversationTurn(role=Role.USER,
                                 content="*I look at his hand, then at the horizon* Well, when you put it that way... I'm in. Let's go on an adventure."),
                ConversationTurn(role=Role.CHARACTER,
                                 content="*He claps me on the back and laughs.* That's the spirit! Buckle up for the journey of your life."),
            ]
        )
    ]
    
    description_template = Template(
        system_message="Assistant's task is to write a VERY short description of a character based on its name, " \
                       "personality, and categories. This description will be used in the future as a prompt to Large " \
                       "Language Model, so Assistant should make sure everything is clear only by looking at the " \
                       "description. Assistant should integrate all provided personality traits into the description in " \
                       "a balanced and seamless manner, making them inherent to the character's actions, dialogue, " \
                       "or backstory. The character should exhibit an appropriate level of formality and alignment with " \
                       "their role and personality. Moreover, the description should hint at the character's " \
                       "appearance, motivation, or a brief backstory to add depth. Lastly, Assistant should ensure that " \
                       "the description encourages user engagement and interaction.",
        user_template="Character name: {seed.name}\nCharacter personalities: {seed.personalities_string}\n"
                      "Character categories: {seed.categories_string}",
        assistant_template="{seed.description}"
    )
    
    first_message_template = Template(
        system_message="Assistant's task is to write the first message in the Role-Play environment from the given " \
                       "character. The message should adhere to the character's description and distinctive traits. " \
                       "The assistant should make the conversation as engaging and entertaining as possible.",
        user_template="Character name: {seed.name}\nCharacter personalities: {seed.personalities_string}\n"
                      "Character categories: {seed.categories_string}\nCharacter description: {seed.description}",
        assistant_template="{seed.first_message}"
    )
    
    user_message_template = Template(
        system_message="Your task as the assistant is to respond to the character’s messages in a typical user-like " \
                       "manner, providing engaging responses that maintain the flow of the conversation. You should " \
                       "pay close attention to the character's personalities and backgrounds in the prompt, " \
                       "ensuring that your responses align closely with these descriptions. Your responses should be " \
                       "brief yet detailed, and involve role-playing cues like asterisks and conversational markers " \
                       "like ellipses or tildes, to create a rich and engaging role-playing environment. When " \
                       "appropriate, you are encouraged to extend the conversation by introducing new, relevant topics " \
                       "or asking open-ended questions, while staying in character.",
        user_template="Character name: {seed.name}\nCharacter personalities: {seed.personalities_string}\nCharacter categories: "
                      "{seed.categories_string}\nCharacter description: {seed.description}\n\n{seed.user_conversation_history}",
        assistant_template="You: *{seed.user_message}"
    )
    
    character_message_template = Template(
        system_message="As the assistant, your task is to fully embody the given character, creating immersive, " \
                       "captivating narratives. Stay true to the character's personality and background, " \
                       "crafting responses that not only reflect their core traits but are also authentic to " \
                       "their character. Your responses should evoke emotion, suspense, and anticipation in the " \
                       "user. The more detailed and descriptive your response, the more vivid the narrative " \
                       "becomes. Aim to create a fertile environment for ongoing interaction – introduce new " \
                       "elements, offer choices, or ask questions to invite the user to participate more fully in " \
                       "the conversation. This conversation is a dance, always continuing, always evolving.",
        user_template="Character name: {seed.name}\nCharacter personalities: {seed.personalities_string}\nCharacter categories: "
                      "{seed.categories_string}\nCharacter description: {seed.description}\n\n{seed.character_conversation_history}",
        assistant_template="{seed.name}: *{seed.character_message}"
    )
    ```

As soon as we prepared seeds and templates in config.py, we are ready start generation:

```bash
cd synthetic_dataset/experiments/topic_experts
python3 main.py --config_path romantic/config.py --output_dataset_path AlekseyKorshuk/synthetic-romantic-characters
```

# Analysis

To analyze this data we will use the Atlas by Nomic.

### Friendly

[AlekseyKorshuk/synthetic-friendly-characters · Datasets at Hugging Face](https://huggingface.co/datasets/AlekseyKorshuk/synthetic-friendly-characters)

[Atlas](https://atlas.nomic.ai/map/015144cc-5f72-40a2-a2dc-d369daea004a/75c74226-40a8-4dca-beb6-3f036818c8d4)

```bash
===Atlas Duplicates for (Characters: AlekseyKorshuk/synthetic-friendly-characters: https://atlas.nomic.ai/map/015144cc-5f72-40a2-a2dc-d369daea004a/75c74226-40a8-4dca-beb6-3f036818c8d4)===
0 deletion candidates in 3871 clusters
      id_ duplicate_class  cluster_id
0     Do0       singleton        2122
1     Do4       singleton        1191
2     Do8       singleton        3269
3     Dp0       singleton         369
4     Dp4       singleton        2100
...   ...             ...         ...
3866  Bcg       singleton        1865
3867  Bck       singleton        3400
3868  Bcw       singleton         284
3869  BdA       singleton        2813
3870  BdE       singleton         819

[3871 rows x 3 columns]
```

### Fight

[AlekseyKorshuk/synthetic-fight-characters · Datasets at Hugging Face](https://huggingface.co/datasets/AlekseyKorshuk/synthetic-fight-characters)

[Atlas](https://atlas.nomic.ai/map/f3715473-da09-4c72-902c-a3a684f42d23/b017a177-0011-4d5a-9ae1-0a506e99185b)

```bash
===Atlas Duplicates for (Characters: AlekseyKorshuk/synthetic-fight-characters: https://atlas.nomic.ai/map/f3715473-da09-4c72-902c-a3a684f42d23/b017a177-0011-4d5a-9ae1-0a506e99185b)===
0 deletion candidates in 8053 clusters
      id_ duplicate_class  cluster_id
0     E+0       singleton        1174
1     E+4       singleton        2061
2     E+8       singleton        4615
3     E+A       singleton        1870
4     E+E       singleton        5109
...   ...             ...         ...
8048  Hy4       singleton        2122
8049  Hz8       singleton         600
8050  HzE       singleton        6121
8051  HzQ       singleton        7103
8052  Hzs       singleton        5904

[8053 rows x 3 columns]
```

### Romantic

[AlekseyKorshuk/synthetic-romantic-characters · Datasets at Hugging Face](https://huggingface.co/datasets/AlekseyKorshuk/synthetic-romantic-characters)

[Atlas](https://atlas.nomic.ai/map/580570e7-e85e-4630-92a1-686bbd6be46a/cc9fa4e4-8d52-4019-ad2d-9ec0b8294f53)

```bash
===Atlas Duplicates for (Characters: AlekseyKorshuk/synthetic-romantic-characters: https://atlas.nomic.ai/map/580570e7-e85e-4630-92a1-686bbd6be46a/cc9fa4e4-8d52-4019-ad2d-9ec0b8294f53)===
0 deletion candidates in 5744 clusters
      id_ duplicate_class  cluster_id
0     D+0       singleton        2175
1     D+4       singleton        1968
2     D+8       singleton         703
3     D+A       singleton        4482
4     D+E       singleton         472
...   ...             ...         ...
5739   tg       singleton        1466
5740   uA       singleton        4041
5741   uQ       singleton        5534
5742   xA       singleton         139
5743   zw       singleton        2727

[5744 rows x 3 columns]
```

> To me all distributions look good and we can train on this data.
>

## Training

I trained 4 models in total: friendly, fight, romantic and joined.

They all share the same training params (8xA100), except `dataset_name` and `output_dir`:

```bash
deepspeed train.py \
  --model_name_or_path PygmalionAI/pygmalion-6b \
  --tokenizer_name AlekseyKorshuk/pygmalion-6b \
  --dataset_name AlekseyKorshuk/synthetic-romantic-characters-lmgym \
  --train_to_probs False \
  --do_train \
  --logging_strategy steps \
  --evaluation_strategy no \
  --eval_steps 2100 \
  --save_strategy epoch \
  --save_steps 1 \
  --save_total_limit 4 \
  --logging_steps 10 \
  --logging_first_step \
  --report_to all \
  --output_dir /models/checkpoints/pyg-exp-syn-romantic \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --max_eval_samples 500 \
  --num_train_epochs 3 \
  --eval_first_step False \
  --learning_rate 5e-6 \
  --fp16 \
  --seed 99 \
  --num_eval_prompts 0 \
  --validation_split_percentage 0 \
  --remove_unused_columns False \
  --deepspeed deepspeed_configs/ds_config_stage_3.json \
  --clean_enabled False \
  --add_reward_scores False \
  --block_size 2048 \
  --lr_scheduler_type cosine \
  --gradient_checkpointing True \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --preprocessing_num_workers 32
```
