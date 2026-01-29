from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for using session

# Load trained model
model = tf.keras.models.load_model('resnet_model_finetuned2.h5')

# Define class labels
class_labels = ["antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat", 
    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", 
    "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", 
    "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", 
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", "koala", 
    "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus", 
    "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", 
    "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros", 
    "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel", 
    "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"]

# Animal information dictionary
animal_info = {
    "antelope": {
        "description": "Antelopes are herbivorous mammals known for their agility and speed.",
        "habitat": "Grasslands and savannas",
        "diet": "Herbivore",
        "lifespan": "10-25 years",
        "image": "https://example.com/antelope.jpg"
    },
    "badger": {
        "description": "Badgers are burrowing mammals with strong claws and a distinctive black-and-white face.",
        "habitat": "Forests, grasslands, and meadows",
        "diet": "Omnivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/badger.jpg"
    },
    "bat": {
        "description": "Bats are the only flying mammals, known for echolocation and nocturnal habits.",
        "habitat": "Caves, forests, and urban areas",
        "diet": "Varies (insectivores, frugivores, or nectarivores)",
        "lifespan": "5-30 years",
        "image": "https://example.com/bat.jpg"
    },
    "bear": {
        "description": "Bears are large mammals known for their strength and adaptability.",
        "habitat": "Forests, mountains, and tundras",
        "diet": "Omnivore",
        "lifespan": "20-30 years",
        "image": "https://example.com/bear.jpg"
    },
    "bee": {
        "description": "Bees are flying insects known for pollination and honey production.",
        "habitat": "Forests, gardens, and meadows",
        "diet": "Nectar and pollen",
        "lifespan": "A few weeks to several months",
        "image": "https://example.com/bee.jpg"
    },
    "beetle": {
        "description": "Beetles are insects with hard exoskeletons and diverse adaptations.",
        "habitat": "Forests, grasslands, and urban areas",
        "diet": "Varies (herbivores, carnivores, decomposers)",
        "lifespan": "A few months to several years",
        "image": "https://example.com/beetle.jpg"
    },
    "bison": {
        "description": "Bison are large herbivorous mammals known for their shaggy coats and strong build.",
        "habitat": "Grasslands and prairies",
        "diet": "Herbivore",
        "lifespan": "15-20 years",
        "image": "https://example.com/bison.jpg"
    },
    "boar": {
        "description": "Boars are wild relatives of domestic pigs, known for their tusks and aggressive nature.",
        "habitat": "Forests and grasslands",
        "diet": "Omnivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/boar.jpg"
    },
    "butterfly": {
        "description": "Butterflies are insects with colorful wings and a metamorphic life cycle.",
        "habitat": "Forests, meadows, and gardens",
        "diet": "Nectar (adult), leaves (caterpillar)",
        "lifespan": "A few weeks to several months",
        "image": "https://example.com/butterfly.jpg"
    },
    "cat": {
        "description": "Cats are small carnivorous mammals often kept as pets.",
        "habitat": "Domestic, forests, and urban areas",
        "diet": "Carnivore",
        "lifespan": "12-18 years",
        "image": "https://example.com/cat.jpg"
    },
    "caterpillar": {
        "description": "Caterpillars are the larval stage of butterflies and moths.",
        "habitat": "Forests, gardens, and meadows",
        "diet": "Herbivore",
        "lifespan": "A few weeks before metamorphosis",
        "image": "https://example.com/caterpillar.jpg"
    },
    "chimpanzee": {
        "description": "Chimpanzees are highly intelligent primates closely related to humans.",
        "habitat": "Rainforests and woodlands",
        "diet": "Omnivore",
        "lifespan": "30-50 years",
        "image": "https://example.com/chimpanzee.jpg"
    },
    "cockroach": {
        "description": "Cockroaches are resilient insects known for their adaptability.",
        "habitat": "Urban areas, forests, and humid places",
        "diet": "Omnivore",
        "lifespan": "A few months to a couple of years",
        "image": "https://example.com/cockroach.jpg"
    },
    "cow": {
        "description": "Cows are domesticated herbivores known for milk and meat production.",
        "habitat": "Farms and grasslands",
        "diet": "Herbivore",
        "lifespan": "15-20 years",
        "image": "https://example.com/cow.jpg"
    },
    "coyote": {
        "description": "Coyotes are wild canines known for their adaptability and cunning behavior.",
        "habitat": "Forests, deserts, and urban areas",
        "diet": "Omnivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/coyote.jpg"
    },
    "crab": {
        "description": "Crabs are crustaceans known for their hard shells and sideways walking.",
        "habitat": "Coastal areas and oceans",
        "diet": "Omnivore",
        "lifespan": "3-30 years",
        "image": "https://example.com/crab.jpg"
    },
    "crow": {
        "description": "Crows are highly intelligent birds known for problem-solving abilities.",
        "habitat": "Urban areas, forests, and fields",
        "diet": "Omnivore",
        "lifespan": "7-15 years",
        "image": "https://example.com/crow.jpg"
    },
    "deer": {
        "description": "Deer are graceful herbivorous mammals with antlers in males.",
        "habitat": "Forests and grasslands",
        "diet": "Herbivore",
        "lifespan": "10-20 years",
        "image": "https://example.com/deer.jpg"
    },
    "dog": {
        "description": "Dogs are domesticated mammals known for loyalty and companionship.",
        "habitat": "Domestic and wild",
        "diet": "Omnivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/dog.jpg"
    },
    "dolphin": {
        "description": "Dolphins are highly intelligent marine mammals known for social behavior.",
        "habitat": "Oceans and seas",
        "diet": "Carnivore",
        "lifespan": "30-50 years",
        "image": "https://example.com/dolphin.jpg"
    },
    "donkey": {
        "description": "Donkeys are domesticated animals used for transportation and labor.",
        "habitat": "Farms and deserts",
        "diet": "Herbivore",
        "lifespan": "25-30 years",
        "image": "https://example.com/donkey.jpg"
    },
    "dragonfly": {
        "description": "Dragonflies are fast-flying insects with large wings and excellent vision.",
        "habitat": "Near water bodies like ponds and lakes",
        "diet": "Carnivore (feeds on insects)",
        "lifespan": "6 months to a few years",
        "image": "https://example.com/dragonfly.jpg"
    },
    "duck": {
        "description": "Ducks are waterfowl known for their webbed feet and varied colors.",
        "habitat": "Lakes, rivers, and wetlands",
        "diet": "Omnivore",
        "lifespan": "5-10 years",
        "image": "https://example.com/duck.jpg"
    },
    "eagle": {
        "description": "Eagles are birds of prey with sharp talons and excellent eyesight.",
        "habitat": "Mountains, forests, and coastal regions",
        "diet": "Carnivore",
        "lifespan": "20-30 years",
        "image": "https://example.com/eagle.jpg"
    },
    "flamingo": {
        "description": "Flamingos are wading birds known for their pink feathers and long legs.",
        "habitat": "Lakes, lagoons, and wetlands",
        "diet": "Omnivore (algae, crustaceans)",
        "lifespan": "20-30 years",
        "image": "https://example.com/flamingo.jpg"
    },
    "fly": {
        "description": "Flies are small winged insects known for rapid reproduction.",
        "habitat": "Urban areas, forests, and farms",
        "diet": "Varied (depends on species)",
        "lifespan": "A few weeks",
        "image": "https://example.com/fly.jpg"
    },
    "fox": {
        "description": "Foxes are small, omnivorous mammals with bushy tails and keen senses.",
        "habitat": "Forests, grasslands, and urban areas",
        "diet": "Omnivore",
        "lifespan": "3-10 years",
        "image": "https://example.com/fox.jpg"
    },
    "goat": {
        "description": "Goats are domesticated animals known for their agility and milk production.",
        "habitat": "Farms, mountains, and grasslands",
        "diet": "Herbivore",
        "lifespan": "10-18 years",
        "image": "https://example.com/goat.jpg"
    },
    "goldfish": {
        "description": "Goldfish are small freshwater fish commonly kept as pets.",
        "habitat": "Ponds and aquariums",
        "diet": "Omnivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/goldfish.jpg"
    },
    "goose": {
        "description": "Geese are large waterfowl known for their honking calls and migration patterns.",
        "habitat": "Lakes, ponds, and fields",
        "diet": "Herbivore",
        "lifespan": "10-25 years",
        "image": "https://example.com/goose.jpg"
    },
    "gorilla": {
        "description": "Gorillas are large primates known for their intelligence and strength.",
        "habitat": "Rainforests",
        "diet": "Herbivore",
        "lifespan": "35-50 years",
        "image": "https://example.com/gorilla.jpg"
    },
    "grasshopper": {
        "description": "Grasshoppers are jumping insects known for their strong hind legs.",
        "habitat": "Grasslands and forests",
        "diet": "Herbivore",
        "lifespan": "A few months",
        "image": "https://example.com/grasshopper.jpg"
    },
    "hamster": {
        "description": "Hamsters are small rodents often kept as pets.",
        "habitat": "Deserts and grasslands",
        "diet": "Omnivore",
        "lifespan": "2-3 years",
        "image": "https://example.com/hamster.jpg"
    },
    "hare": {
        "description": "Hares are fast-moving mammals similar to rabbits but with longer legs.",
        "habitat": "Grasslands and woodlands",
        "diet": "Herbivore",
        "lifespan": "4-8 years",
        "image": "https://example.com/hare.jpg"
    },
    "hedgehog": {
        "description": "Hedgehogs are small, spiny mammals known for their nocturnal behavior.",
        "habitat": "Forests, gardens, and grasslands",
        "diet": "Omnivore",
        "lifespan": "3-7 years",
        "image": "https://example.com/hedgehog.jpg"
    },
    "hippopotamus": {
        "description": "Hippopotamuses are large semi-aquatic mammals known for their size and aggression.",
        "habitat": "Rivers and lakes",
        "diet": "Herbivore",
        "lifespan": "40-50 years",
        "image": "https://example.com/hippopotamus.jpg"
    },
    "hornbill": {
        "description": "Hornbills are birds with large, curved beaks and striking features.",
        "habitat": "Forests and tropical regions",
        "diet": "Omnivore",
        "lifespan": "20-40 years",
        "image": "https://example.com/hornbill.jpg"
    },
    "horse": {
        "description": "Horses are domesticated animals known for their speed and strength.",
        "habitat": "Farms, grasslands, and mountains",
        "diet": "Herbivore",
        "lifespan": "25-30 years",
        "image": "https://example.com/horse.jpg"
    },
    "hummingbird": {
        "description": "Hummingbirds are small, fast birds known for hovering flight and drinking nectar.",
        "habitat": "Forests and gardens",
        "diet": "Nectar and small insects",
        "lifespan": "3-5 years",
        "image": "https://example.com/hummingbird.jpg"
    },
    "hyena": {
        "description": "Hyenas are carnivorous mammals known for their laughter-like calls and strong bite.",
        "habitat": "Savannas and forests",
        "diet": "Carnivore",
        "lifespan": "10-25 years",
        "image": "https://example.com/hyena.jpg"
    },
    "jellyfish": {
        "description": "Jellyfish are gelatinous marine animals with tentacles that can sting.",
        "habitat": "Oceans",
        "diet": "Carnivore",
        "lifespan": "A few months to several years",
        "image": "https://example.com/jellyfish.jpg"
    },
    "kangaroo": {
        "description": "Kangaroos are marsupials known for their strong hind legs and pouches.",
        "habitat": "Grasslands and forests",
        "diet": "Herbivore",
        "lifespan": "6-20 years",
        "image": "https://example.com/kangaroo.jpg"
    },
    "koala": {
        "description": "Koalas are tree-dwelling marsupials native to Australia.",
        "habitat": "Eucalyptus forests",
        "diet": "Herbivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/koala.jpg"
    },
    "ladybugs": {
        "description": "Ladybugs are small beetles known for their bright red shells with black spots.",
        "habitat": "Gardens, forests, and grasslands",
        "diet": "Carnivore (aphids and small insects)",
        "lifespan": "1-2 years",
        "image": "https://example.com/ladybugs.jpg"
    },
    "leopard": {
        "description": "Leopards are powerful big cats known for their spotted coats and stealth.",
        "habitat": "Forests, savannas, and mountains",
        "diet": "Carnivore",
        "lifespan": "12-17 years",
        "image": "https://example.com/leopard.jpg"
    },
    "lion": {
        "description": "Lions are large carnivorous cats known as the 'King of the Jungle'.",
        "habitat": "Grasslands and savannas",
        "diet": "Carnivore",
        "lifespan": "10-14 years",
        "image": "https://example.com/lion.jpg"
    },
    "lizard": {
        "description": "Lizards are cold-blooded reptiles with scaly skin and long tails.",
        "habitat": "Deserts, forests, and grasslands",
        "diet": "Varies (insects, plants, small animals)",
        "lifespan": "1-30 years (depending on species)",
        "image": "https://example.com/lizard.jpg"
    },
    "lobster": {
        "description": "Lobsters are marine crustaceans with hard shells and strong claws.",
        "habitat": "Oceans and sea floors",
        "diet": "Omnivore",
        "lifespan": "40-60 years",
        "image": "https://example.com/lobster.jpg"
    },
    "mosquito": {
        "description": "Mosquitoes are small flying insects known for feeding on blood.",
        "habitat": "Wetlands, forests, and urban areas",
        "diet": "Herbivore (nectar), females feed on blood",
        "lifespan": "1-2 weeks",
        "image": "https://example.com/mosquito.jpg"
    },
    "moth": {
        "description": "Moths are nocturnal insects closely related to butterflies.",
        "habitat": "Forests, meadows, and homes",
        "diet": "Herbivore (nectar, leaves)",
        "lifespan": "A few days to several months",
        "image": "https://example.com/moth.jpg"
    },
    "mouse": {
        "description": "Mice are small rodents known for their adaptability and quick reproduction.",
        "habitat": "Fields, forests, and homes",
        "diet": "Omnivore",
        "lifespan": "1-3 years",
        "image": "https://example.com/mouse.jpg"
    },
    "octopus": {
        "description": "Octopuses are intelligent marine creatures with eight arms and a soft body.",
        "habitat": "Oceans and coral reefs",
        "diet": "Carnivore",
        "lifespan": "1-5 years",
        "image": "https://example.com/octopus.jpg"
    },
    "okapi": {
        "description": "Okapis are rare, forest-dwelling relatives of giraffes.",
        "habitat": "Dense rainforests",
        "diet": "Herbivore",
        "lifespan": "20-30 years",
        "image": "https://example.com/okapi.jpg"
    },
    "orangutan": {
        "description": "Orangutans are large, intelligent primates known for their tree-dwelling lifestyle.",
        "habitat": "Tropical rainforests",
        "diet": "Omnivore (fruits, leaves, insects)",
        "lifespan": "30-40 years",
        "image": "https://example.com/orangutan.jpg"
    },
    "otter": {
        "description": "Otters are playful, semi-aquatic mammals known for their agility in water.",
        "habitat": "Rivers, lakes, and coastal areas",
        "diet": "Carnivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/otter.jpg"
    },
    "owl": {
        "description": "Owls are nocturnal birds of prey known for their excellent night vision.",
        "habitat": "Forests, mountains, and grasslands",
        "diet": "Carnivore",
        "lifespan": "5-15 years",
        "image": "https://example.com/owl.jpg"
    },
    "ox": {
        "description": "Oxen are domesticated cattle known for their strength and endurance.",
        "habitat": "Farms and grasslands",
        "diet": "Herbivore",
        "lifespan": "15-20 years",
        "image": "https://example.com/ox.jpg"
    },
    "oyster": {
        "description": "Oysters are marine mollusks known for producing pearls.",
        "habitat": "Oceans and coastal areas",
        "diet": "Filter feeder",
        "lifespan": "10-20 years",
        "image": "https://example.com/oyster.jpg"
    },
    "panda": {
        "description": "Giant pandas are bears known for their black-and-white fur and bamboo diet.",
        "habitat": "Mountain forests",
        "diet": "Herbivore (mainly bamboo)",
        "lifespan": "20-30 years",
        "image": "https://example.com/panda.jpg"
    },
    "parrot": {
        "description": "Parrots are colorful, intelligent birds known for their ability to mimic sounds.",
        "habitat": "Tropical forests and savannas",
        "diet": "Herbivore",
        "lifespan": "20-80 years",
        "image": "https://example.com/parrot.jpg"
    },
    "pelecaniformes": {
        "description": "Pelecaniformes are a group of large water birds including pelicans and cormorants.",
        "habitat": "Wetlands, lakes, and coastal regions",
        "diet": "Carnivore (fish, crustaceans)",
        "lifespan": "10-25 years",
        "image": "https://example.com/pelecaniformes.jpg"
    },
    "penguin": {
        "description": "Penguins are flightless birds adapted for swimming in cold waters.",
        "habitat": "Antarctic and coastal regions",
        "diet": "Carnivore (fish, krill)",
        "lifespan": "15-20 years",
        "image": "https://example.com/penguin.jpg"
    },
    "pig": {
        "description": "Pigs are highly intelligent, social animals raised for food production.",
        "habitat": "Farms and forests",
        "diet": "Omnivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/pig.jpg"
    },
    "pigeon": {
        "description": "Pigeons are adaptable birds known for their homing abilities.",
        "habitat": "Urban areas and forests",
        "diet": "Herbivore",
        "lifespan": "3-5 years",
        "image": "https://example.com/pigeon.jpg"
    },
    "porcupine": {
        "description": "Porcupines are rodents covered in sharp quills for protection.",
        "habitat": "Forests, deserts, and grasslands",
        "diet": "Herbivore",
        "lifespan": "5-7 years",
        "image": "https://example.com/porcupine.jpg"
    },
    "possum": {
        "description": "Possums are nocturnal marsupials known for playing dead as a defense mechanism.",
        "habitat": "Forests and urban areas",
        "diet": "Omnivore",
        "lifespan": "4-6 years",
        "image": "https://example.com/possum.jpg"
    },
    "raccoon": {
        "description": "Raccoons are nocturnal mammals known for their intelligence and dexterous paws.",
        "habitat": "Forests, urban areas, and wetlands",
        "diet": "Omnivore",
        "lifespan": "2-5 years (wild), 10-15 years (captivity)",
        "image": "https://example.com/raccoon.jpg"
    },
    "rat": {
        "description": "Rats are small, intelligent rodents known for their adaptability and problem-solving skills.",
        "habitat": "Urban areas, forests, and fields",
        "diet": "Omnivore",
        "lifespan": "1-3 years",
        "image": "https://example.com/rat.jpg"
    },
    "reindeer": {
        "description": "Reindeer, also known as caribou in North America, are large deer adapted for cold climates.",
        "habitat": "Arctic tundras and boreal forests",
        "diet": "Herbivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/reindeer.jpg"
    },
    "rhinoceros": {
        "description": "Rhinoceroses are large, thick-skinned herbivores known for their distinctive horns.",
        "habitat": "Grasslands and savannas",
        "diet": "Herbivore",
        "lifespan": "35-50 years",
        "image": "https://example.com/rhinoceros.jpg"
    },
    "sandpiper": {
        "description": "Sandpipers are small shorebirds known for their quick movements along beaches and wetlands.",
        "habitat": "Coastal regions and wetlands",
        "diet": "Carnivore (insects, crustaceans)",
        "lifespan": "5-10 years",
        "image": "https://example.com/sandpiper.jpg"
    },
    "seahorse": {
        "description": "Seahorses are unique marine fish known for their horse-like head and upright swimming posture.",
        "habitat": "Shallow coastal waters and coral reefs",
        "diet": "Carnivore (small crustaceans)",
        "lifespan": "1-5 years",
        "image": "https://example.com/seahorse.jpg"
    },
    "seal": {
        "description": "Seals are semi-aquatic marine mammals known for their streamlined bodies and flippers.",
        "habitat": "Coastal waters and ice-covered regions",
        "diet": "Carnivore (fish, squid)",
        "lifespan": "15-30 years",
        "image": "https://example.com/seal.jpg"
    },
    "shark": {
        "description": "Sharks are powerful predatory fish with cartilaginous skeletons and sharp teeth.",
        "habitat": "Oceans worldwide",
        "diet": "Carnivore",
        "lifespan": "20-70 years",
        "image": "https://example.com/shark.jpg"
    },
    "sheep": {
        "description": "Sheep are domesticated herbivores known for their wool and docile nature.",
        "habitat": "Grasslands and farms",
        "diet": "Herbivore",
        "lifespan": "10-12 years",
        "image": "https://example.com/sheep.jpg"
    },
    "snake": {
        "description": "Snakes are legless reptiles known for their flexible bodies and some species' venomous bites.",
        "habitat": "Forests, deserts, and wetlands",
        "diet": "Carnivore",
        "lifespan": "5-30 years",
        "image": "https://example.com/snake.jpg"
    },
    "sparrow": {
        "description": "Sparrows are small, social birds found in urban and rural areas worldwide.",
        "habitat": "Cities, forests, and grasslands",
        "diet": "Omnivore",
        "lifespan": "3-5 years",
        "image": "https://example.com/sparrow.jpg"
    },
    "squid": {
        "description": "Squids are intelligent marine mollusks known for their ink defense mechanism and fast swimming.",
        "habitat": "Oceans and deep-sea waters",
        "diet": "Carnivore",
        "lifespan": "1-5 years",
        "image": "https://example.com/squid.jpg"
    },
    "squirrel": {
        "description": "Squirrels are agile rodents known for their bushy tails and tree-climbing abilities.",
        "habitat": "Forests, parks, and urban areas",
        "diet": "Omnivore",
        "lifespan": "6-12 years",
        "image": "https://example.com/squirrel.jpg"
    },
    "starfish": {
        "description": "Starfish are marine echinoderms known for their five-arm symmetry and regeneration ability.",
        "habitat": "Oceans and coral reefs",
        "diet": "Carnivore (mollusks, coral)",
        "lifespan": "5-35 years",
        "image": "https://example.com/starfish.jpg"
    },
    "swan": {
        "description": "Swans are large water birds known for their graceful appearance and strong pair bonds.",
        "habitat": "Lakes, rivers, and wetlands",
        "diet": "Herbivore",
        "lifespan": "20-30 years",
        "image": "https://example.com/swan.jpg"
    },
    "tiger": {
        "description": "Tigers are the largest cat species, known for their powerful build and striped coat.",
        "habitat": "Forests and grasslands",
        "diet": "Carnivore",
        "lifespan": "10-15 years",
        "image": "https://example.com/tiger.jpg"
    },
    "turkey": {
        "description": "Turkeys are large ground birds known for their distinctive gobble and fan-shaped tails.",
        "habitat": "Forests and grasslands",
        "diet": "Omnivore",
        "lifespan": "3-10 years",
        "image": "https://example.com/turkey.jpg"
    },
    "turtle": {
        "description": "Turtles are reptiles with protective shells and long lifespans.",
        "habitat": "Oceans, rivers, and forests",
        "diet": "Omnivore",
        "lifespan": "50-100 years",
        "image": "https://example.com/turtle.jpg"
    },
    "whale": {
        "description": "Whales are the largest marine mammals, known for their intelligence and complex songs.",
        "habitat": "Oceans worldwide",
        "diet": "Carnivore (krill, fish)",
        "lifespan": "30-90 years",
        "image": "https://example.com/whale.jpg"
    },
    "wolf": {
        "description": "Wolves are social carnivores known for living in packs and their strong hunting skills.",
        "habitat": "Forests, tundras, and mountains",
        "diet": "Carnivore",
        "lifespan": "6-13 years",
        "image": "https://example.com/wolf.jpg"
    },
    "wombat": {
        "description": "Wombats are burrowing marsupials native to Australia, known for their strong claws.",
        "habitat": "Forests and grasslands",
        "diet": "Herbivore",
        "lifespan": "5-15 years",
        "image": "https://example.com/wombat.jpg"
    },
    "woodpecker": {
        "description": "Woodpeckers are birds known for their strong beaks, which they use to peck trees in search of food.",
        "habitat": "Forests and woodlands",
        "diet": "Omnivore",
        "lifespan": "4-12 years",
        "image": "https://example.com/woodpecker.jpg"
    },
    "zebra": {
        "description": "Zebras are herbivorous mammals known for their unique black-and-white striped coats.",
        "habitat": "Grasslands and savannas",
        "diet": "Herbivore",
        "lifespan": "20-30 years",
        "image": "https://example.com/zebra.jpg"
    }

}

# More animals can be added in the same format...




def preprocess_image(image):
    """Preprocess image before making predictions"""
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return render_template("index.html")
    

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]  # Get predicted class

    # Get animal info
    animal_data = animal_info.get(predicted_class, {
        "description": "No information available.",
        "habitat": "Unknown",
        "diet": "Unknown",
        "lifespan": "Unknown",
        "image": "https://example.com/default.jpg"
    })

    # Store result in session
    session['animal'] = predicted_class
    session['info'] = animal_data

    return redirect(url_for('result'))

@app.route('/result', methods=['GET'])
def result():
    """Render the result page with prediction data"""
    if 'animal' not in session:
        return redirect(url_for('home'))  # Redirect if no result available

    return render_template("result.html", animal=session['animal'], info=session['info'])

if __name__ == '__main__':
    app.run(debug=True)
