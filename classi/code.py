def how_generator_work(self, datagen, X, ID):
    generator = datagen.flow(X, ID, batch_size=64, shuffle=True)
    # Iterator restituisce un batch di immagini per ogni iterazione
    i = 0
    for X_batch, Y_batch in generator:
        img_batch = X_batch
        # ID_batch = Y_batch
        # print(img_batch.shape)
        # print(ID_batch)

        plt.figure(i, figsize=(12, 12))
         for idx in range(img_batch.shape[0]):
            plt.subplot(8, 8, idx + 1)
            plt.axis('off')
            image = img_batch[idx]
            plt.imshow(array_to_img(image), cmap='gray')
        i += 1
        if i % 4 == 0:
            break
    plt.tight_layout()
    plt.show()