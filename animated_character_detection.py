import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
import io

# Cache the model and processor to ensure they are loaded only once
@st.cache_resource
def load_model():
     try:
        # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to("cpu")
        # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
     except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model and processor only once
model, processor = load_model()

# Define animated character categories
animated_characters = [
    # "Mickey Mouse", "SpongeBob", "Naruto", "Pikachu", "Goku", "Elsa (Frozen)", 
    # "Animated Character", "Miles Morales", "Daffy Duck", "Lois Griffin"
    "Mickey Mouse 🐭",
"Donald Duck 🦆",
"SpongeBob SquarePants 🍍",
"Bugs Bunny 🥕",
"Homer Simpson 🍩",
"Bart Simpson 🎸",
"Scooby-Doo 🐶",
"Shrek 🐸",
"Sonic the Hedgehog 🦔",
"Tom and Jerry 🐱🐭",
"Pikachu ⚡",
"Optimus Prime 🚗",
"Woody (Toy Story) 🤠",
"Buzz Lightyear 🚀",
"Daffy Duck 🦆",
"Fred Flintstone 🦕",
"George Jetson 👨‍🚀",
"Winnie the Pooh 🍯",
"Tigger 🐅",
"Elsa (Frozen) ❄️",
"Anna (Frozen) 👸",
"Mufasa (The Lion King) 🦁",
"Buzz Lightyear 🚀",
"Lightning McQueen 🚗",
"Bambi 🦌",
"Dumbo 🐘",
"Stitch (Lilo & Stitch) 👽",
"Timon and Pumbaa 🦁🐗",
"Peter Pan 🧚",
"Ariel (The Little Mermaid) 🧜‍♀️",
"Cinderella 👠",
"Rapunzel (Tangled) 🌸",
"Belle (Beauty and the Beast) 📚",
"Olaf (Frozen) ⛄",
"Scar (The Lion King) 🦁",
"Jack Sparrow (Pirates of the Caribbean) 🏴‍☠️",
"Meg Griffin (Family Guy) 👩",
"Stewie Griffin (Family Guy) 👶",
"South Park Boys (Cartman, Stan, Kyle, Kenny) 🎒",
"Astro Boy 👦",
"Goku (Dragon Ball) 🐉",
"Naruto Uzumaki 🥷",
"Sailor Moon 🌙",
"Ash Ketchum ⚡",
"Velma Dinkley (Scooby-Doo) 🔍",
"Shaggy Rogers (Scooby-Doo) 🍕",
"Kim Possible 🦸‍♀️",
"Ron Stoppable 🍔",
"Zuko (Avatar: The Last Airbender) 🔥",
"Aang (Avatar: The Last Airbender) 🌪️",
"Korra (The Legend of Korra) 🌊",
"Sailor Mars 🔥",
"Felix the Cat 🐱",
"Courage the Cowardly Dog 🐶",
"Popeye the Sailor Man 🥬",
"Babar the Elephant 🐘",
"Hello Kitty 🎀",
"Puss in Boots 🥷",
"Snoopy 🐶",
"Charlie Brown ☁️",
"Lola Bunny 🏀",
"Bugs Bunny 🎺",
"Foghorn Leghorn 🐔",
"Porky Pig 🐖",
"Daria Morgendorffer 📖",
"Beavis 🌪️",
"Butthead 🎸",
"Lilo (Lilo & Stitch) 🌺",
"Dr. Eggman (Sonic the Hedgehog) 🧑‍🔬",
"Yzma (The Emperor's New Groove) 🌌",
"Kronk (The Emperor's New Groove) 🍳",
"WALL-E 🤖",
"EVE 🤖",
"Boo (Monsters, Inc.) 👶",
"Mike Wazowski 👁️",
"Sulley (Monsters, Inc.) 🐻",
"Lightning McQueen 🚗",
"Mater (Cars) 🚜",
"Carl Fredricksen (Up) 🎈",
"Russell (Up) 🦸‍♂️",
"Rex (Toy Story) 🦖",
"Jessie (Toy Story) 🤠",
"Dory (Finding Nemo) 🐠",
"Marlin (Finding Nemo) 🐟",
"Gus Gus (Cinderella) 🐭",
"Gollum (The Lord of the Rings: The Animated Series) 🧙‍♂️",
"Jimmy Neutron 👨‍🔬",
"Dexter (Dexter's Laboratory) 🥼",
"Ed, Edd n Eddy 🍬",
"Timmy Turner (The Fairly OddParents) 👦",
"Cosmo and Wanda 👽🧚‍♀️",
"Mabel Pines (Gravity Falls) 🎀",
"Dipper Pines (Gravity Falls) 📖",
"Steven Universe 🌌",
"Garnet (Steven Universe) 💎",
"Finn the Human (Adventure Time) 🐶",
"Jake the Dog (Adventure Time) 🐕",
"Marceline the Vampire Queen 🎸",
"Danny Phantom 👻",
"Mavis (Hotel Transylvania) 🧛‍♀️",
]

# Function to classify an image
def classify_image(image, categories, confidence_threshold):
    try:
        inputs = processor(text=categories, images=image, return_tensors="pt", padding=True).to("cpu")
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        best_category_idx = probs.argmax().item()
        best_category = categories[best_category_idx]
        confidence = probs[0, best_category_idx].item()
        
        if confidence >= confidence_threshold:
            return best_category, confidence
        else:
            return None, None
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None, None

# Streamlit App UI
st.title("Cartoon Detection AI App 💩👺")
st.write("Upload an image, and the app will detect animated characters!!")

# Confidence threshold slider
confidence_threshold = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.5)

# Image upload
uploaded_files = st.file_uploader("Upload images of animated characters", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            
            # Display original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Perform classification
            category, confidence = classify_image(image, animated_characters, confidence_threshold)
            
            if category:
                st.write(f"Detected Character: {category} (Confidence: {confidence:.2f})")
            else:
                st.write("No character met the confidence threshold.")

        except UnidentifiedImageError:
            st.error("Error: The uploaded file is not a valid image.")
        except Exception as e:
            st.error(f"Unexpected error occurred: {e}")
