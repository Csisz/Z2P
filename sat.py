from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import cv2
import numpy

path ='color_imgs/saved_from_restore/nagyapa_mama_gyuri_v2_coloriZed.png'

img = Image.open(path)
converter = ImageEnhance.Color(img)
img2 = converter.enhance(1.5)
h0, s0, t0 = path.partition('.')
h1,s1,t1 = h0.partition('/')
print(t1)


avg_color_per_row = numpy.average(img2, axis=0)
avg_color = numpy.average(avg_color_per_row, axis=0)

print(avg_color)
plt.imsave('%s_coloriZed_SAT.png'%t1, avg_color)
# img2.show()

# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.title('Original')
# plt.axis('off')

# plt.subplot(1,2,2)
# plt.imshow(img2)
# plt.title('Output (SIGGRAPH 17)')
# plt.axis('off')
# plt.show()