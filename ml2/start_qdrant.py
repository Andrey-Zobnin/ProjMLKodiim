import docker
import logging
import time
import os
import sys

# --- Конфигурация ---
IMAGE_NAME = "qdrant/qdrant:latest"  # Используем последнюю стабильную версию
CONTAINER_NAME = "qdrant_instance" # Имя для контейнера
HTTP_PORT = 6333                   # Порт, который будет доступен на вашем хосте
# GRPC_PORT = 6334                 # gRPC порт (если нужен)
PERSISTENCE_PATH = "./qdrant_data" # Папка на вашем хосте для хранения данных Qdrant
# ---------------------

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_docker_running(client):
    """Проверяет, запущен ли Docker демон."""
    try:
        client.ping()
        return True
    except Exception as e:
        logging.error(f"Не удалось подключиться к Docker демону: {e}")
        logging.error("Пожалуйста, убедитесь, что Docker запущен.")
        return False

def start_qdrant_container():
    """Запускает или создает контейнер Qdrant."""
    try:
        client = docker.from_env()
    except docker.errors.DockerException:
        logging.error("Не удалось инициализировать Docker клиент.")
        logging.error("Убедитесь, что Docker установлен и пользователь имеет права доступа.")
        sys.exit(1)

    if not check_docker_running(client):
        sys.exit(1)

    # Убедимся, что папка для данных существует
    abs_persistence_path = os.path.abspath(PERSISTENCE_PATH)
    if not os.path.exists(abs_persistence_path):
        logging.info(f"Создание папки для хранения данных Qdrant: {abs_persistence_path}")
        try:
            os.makedirs(abs_persistence_path)
        except OSError as e:
            logging.error(f"Не удалось создать папку {abs_persistence_path}: {e}")
            sys.exit(1)

    # Проверяем, существует ли контейнер
    try:
        container = client.containers.get(CONTAINER_NAME)
        logging.info(f"Контейнер '{CONTAINER_NAME}' уже существует.")

        # Проверяем статус контейнера
        if container.status == "running":
            logging.info(f"Контейнер '{CONTAINER_NAME}' уже запущен.")
            # Проверим порты на всякий случай
            host_port = container.attrs['HostConfig']['PortBindings'].get(f'{HTTP_PORT}/tcp')
            if not host_port or int(host_port[0]['HostPort']) != HTTP_PORT:
                 logging.warning(f"Контейнер '{CONTAINER_NAME}' запущен, но порт {HTTP_PORT} может быть проброшен неправильно.")
            return container
        else:
            logging.info(f"Контейнер '{CONTAINER_NAME}' остановлен. Запускаем...")
            try:
                container.start()
                logging.info(f"Контейнер '{CONTAINER_NAME}' успешно запущен.")
                # Дадим немного времени на инициализацию
                time.sleep(5)
                return container
            except docker.errors.APIError as e:
                logging.error(f"Не удалось запустить контейнер '{CONTAINER_NAME}': {e}")
                logging.error("Возможно, порт уже занят или есть другие проблемы.")
                sys.exit(1)

    except docker.errors.NotFound:
        logging.info(f"Контейнер '{CONTAINER_NAME}' не найден. Создаем и запускаем новый...")
        try:
            # Пытаемся скачать последнюю версию образа (если его нет локально)
            logging.info(f"Загрузка образа {IMAGE_NAME}...")
            client.images.pull(IMAGE_NAME)

            # Запускаем контейнер
            container = client.containers.run(
                image=IMAGE_NAME,
                name=CONTAINER_NAME,
                ports={f'{HTTP_PORT}/tcp': HTTP_PORT}, # Проброс HTTP порта
                # ports={f'{HTTP_PORT}/tcp': HTTP_PORT, f'{GRPC_PORT}/tcp': GRPC_PORT}, # Если нужен и gRPC
                volumes={abs_persistence_path: {'bind': '/qdrant/storage', 'mode': 'rw'}}, # Подключаем том для данных
                detach=True, # Запуск в фоновом режиме
                restart_policy={"Name": "unless-stopped"} # Политика перезапуска
            )
            logging.info(f"Контейнер '{CONTAINER_NAME}' успешно создан и запущен (ID: {container.short_id}).")
            logging.info(f"Данные будут сохраняться в: {abs_persistence_path}")
            logging.info(f"HTTP API доступен на http://localhost:{HTTP_PORT}")
            # Дадим немного времени на инициализацию
            time.sleep(5)
            return container
        except docker.errors.APIError as e:
            logging.error(f"Не удалось создать или запустить контейнер '{CONTAINER_NAME}': {e}")
            logging.error("Возможные причины: порт занят, проблемы с томом, ошибка Docker.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Произошла непредвиденная ошибка: {e}")
            sys.exit(1)

if __name__ == "__main__":
    start_qdrant_container()
    logging.info("Скрипт завершил работу. Qdrant должен быть запущен.")