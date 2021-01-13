import os, glob, shutil
from PIL import Image

def image_resize(dir_path):
    class_dir= ['scissor', 'rock', 'paper']

    for label in class_dir:
        image_dir_path= os.path.join(dir_path, label)
        print("이미지 디렉토리 경로: ", image_dir_path)

        images=glob.glob(image_dir_path + "/*.jpg")

        # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
        target_size=(28,28)
        for img in images:
            old_img=Image.open(img)
            new_img=old_img.resize(target_size,Image.ANTIALIAS)
            new_img.save(img,"JPEG")

        print("가위 이미지 resize 완료!")


def downDataToTrain():
    downloaded_dataset_dir= os.getcwd()+'/rock_scissor_paper_data/download_data'
    # down load된 dataset은 사람마다 하나의 폴더로 되어있어서 keras data generator를 사용하기에 부적절 해보인다.
    # train폴더로 copy하는 작업을 한다.

    for pd in os.listdir(downloaded_dataset_dir):
        #사람별
        pd_dir= os.path.join(downloaded_dataset_dir,pd)
        if(os.path.isdir(pd_dir) is False):
            continue
        for cls in os.listdir(pd_dir):
            #rock scissor paper 디렉토리별로 탐색
            #for f in glob.iglob(os.path.join(downloaded_dataset_dir,pd,cls)+'/*.jpg'):
            try:
                cls_dir= os.path.join(downloaded_dataset_dir,pd,cls)
                if(os.path.isdir(cls_dir) is False):
                    continue
                for f in os.listdir(cls_dir):
                    dst_directory= os.getcwd()+'/rock_scissor_paper_data/train/'+cls+'/'+pd+'_'
                    shutil.copyfile(os.path.join(downloaded_dataset_dir,pd,cls)+'/'+f, dst_directory+f)
            except Exception as e:
                print(e)
                continue

image_resize(os.getcwd()+'/rock_scissor_paper_data/test')