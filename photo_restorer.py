import replicate
from dotenv import load_dotenv
load_dotenv

def predict_image(filename):
    output = replicate.run(
    "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
    input={
        "img": open(filename, "rb"),
        "scale": 2,
        "version": "v1.4"
    }
    )
    print(output)