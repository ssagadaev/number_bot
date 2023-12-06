from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Загрузка модели
model = tf.keras.models.load_model('mnist_model.h5')

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Отправь мне изображение с цифрой.')

def preprocess_image(image):
    # Преобразование в формат, подходящий для модели
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    return image

def predict_digit(img):
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)

def handle_photo(update: Update, context: CallbackContext) -> None:
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    image = Image.open(f)

    processed_image = preprocess_image(image)
    digit = predict_digit(processed_image)
    update.message.reply_text(f'Я думаю, это цифра: {digit[0]}')

def main() -> None:
    updater = Updater("6356541708:AAFQkH9XsYG0p-BZqHC01kSBGBzb0bxE784")

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()