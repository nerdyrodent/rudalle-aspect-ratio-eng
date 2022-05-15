import sys
import os
sys.path.insert(0, './rudalle-aspect-ratio')
from rudalle_aspect_ratio import RuDalleAspectRatio, get_rudalle_model
from rudalle import get_vae, get_tokenizer
from rudalle.pipelines import show
from deep_translator import GoogleTranslator # don't forget to "pip install deep-translator" ;)

# Settings for this run
text = 'A Nerdy Rodent'
num_images = 4 # (4)
batch_size = 4 # (4)
top_k = 1024 # (1024)
top_p = 0.975 # (0.975)
image_save_dir = 'rodents'
image_prefix = 'version_1'
aspect_ratio = 32/9 # (32/9), (9/32), etc
cache_dir = './cache'

# Seed is random by default

# Do run
if not os.path.exists(image_save_dir):
    os.mkdir(image_save_dir)

if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

device = 'cuda'
translated = GoogleTranslator(source='auto', target='ru').translate(text)

dalle = get_rudalle_model('Surrealist_XL', fp16=True, device=device, cache_dir=cache_dir)
vae, tokenizer = get_vae(cache_dir=cache_dir).to(device), get_tokenizer(cache_dir=cache_dir)
rudalle_ar = RuDalleAspectRatio(
    dalle=dalle, vae=vae, tokenizer=tokenizer,
    aspect_ratio=aspect_ratio, bs=batch_size, device=device,
)
_, result_pil_images = rudalle_ar.generate_images(translated, top_k, top_p, num_images)

# Save instead of Show
#show(result_pil_images, 1)

# Save each image
print('Saving images...')
for i, img in enumerate(result_pil_images):
    img.save(f'{image_save_dir}/{image_prefix}-{i}.jpg', quality=100)
print('Images saved.')
