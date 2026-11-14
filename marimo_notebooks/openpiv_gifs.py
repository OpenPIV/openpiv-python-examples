import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from openpiv.piv import simple_piv
    import imageio
    import matplotlib.pyplot as plt
    import numpy as np

    return imageio, simple_piv


@app.cell
def _(imageio, simple_piv):
    list_of_gifs = ['01','02','03','04','05','06','07','08','21','22','23','24','41']
    base_path = 'http://www.vsj.jp/~pivstd/gif/'
    base_name = 'image'
    base_ext = '.gif'

    for gif in list_of_gifs: 
        print('Reading ..')
        print(f'{base_path}{base_name}{gif}{base_ext}')
        im = imageio.get_reader(f'{base_path}{base_name}{gif}{base_ext}')
        # print(im)

        images = []
        for frame in im:
            if frame.ndim > 2:
                frame = frame[:,:,0]
            
            images.append(frame)
    
        # images = np.array(images)

        # plt.figure()
        # plt.imshow(np.c_[images[0],images[1]])

        # for I,J in zip(images[:-1],images[1:]):
        #     simple_piv(I,J)

        # let's do only one pair 
        simple_piv(images[0], images[1])
    return


if __name__ == "__main__":
    app.run()
