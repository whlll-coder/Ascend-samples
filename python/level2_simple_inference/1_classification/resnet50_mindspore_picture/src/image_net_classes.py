image_net_classes = [
"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"
]

def get_image_net_class(id):
    if id >= len(image_net_classes):
        return "unknown"
    else:
        return image_net_classes[id]
