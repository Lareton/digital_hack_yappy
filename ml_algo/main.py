from algo_utils import download_video
from ml_algo import get_res_by_uuid
from check_duplicate import model, df, vector_db


def check_video_is_duplicate_by_uuid(uuid_video):
    video_link = f"https://s3.ritm.media/yappy-db-duplicates/{uuid_video}.mp4"
    download_video(video_link, f"{uuid_video}.mp4")
    pred_is_dup, pred_uuid = get_res_by_uuid(df, vector_db, model, uuid_video, 0.3)
    return pred_is_dup, pred_uuid


if __name__ == '__main__':
    print(check_video_is_duplicate_by_uuid("6d3233b6-f8de-49ba-8697-bb30dbf825f7"))
