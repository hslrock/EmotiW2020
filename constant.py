
AFFECTNETPATH="../../../hdd/Dataset/AffectNet/cropped_Annotated"
EMOTIPATH="../../../hdd/Dataset/EmotiW/images"
EMOTIFACEPATH="../../../hdd/Dataset/EmotiW/Face_Cropped"

def show_image(data):
    import matplotlib.pyplot as plt
    input=data[0][0]
    plt.figure(figsize=(16, 16))
    grid_border_size = 2
    grid = utils.make_grid(input)

    plt.imshow((grid.numpy().transpose((1, 2, 0))*0.5)+0.5)
    plt.axis('off')
    plt.ioff()
    plt.show()