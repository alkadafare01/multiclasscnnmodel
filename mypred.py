import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow_core.python.keras.models import load_model

model = load_model('model.h5')

# summarize model
model.summary()
#model.predict()
test_image = image.load_img('data/val/10.1016-slash-j.crad.2020.04.002-b.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)
if result[0][1] == 1:
        print('It is a pneumonia')
elif result[0][2] == 1:
    print('It is a covid')
elif result[0][0] == 1:
    print('It is a Normal')