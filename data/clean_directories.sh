#!/bin/bash

DIR_LIST_TXT="all.txt"
#DIR_LIST_TXT="train.txt"

FILE_REMOVE_LIST=("door_features.mat" "neg_door_radial.mat" "pos_door_radial.mat")

echo "Cleaning directories listed in: $DIR_LIST_TXT"
echo 
echo "This will remove the following files: "
for name in "${FILE_REMOVE_LIST[@]}"; do
  echo "> $name"
done
echo

 
echo ">>> [Y] Proceed; [X] Abort"
read response

if [[ $response == [yY] ]]; then
  echo "Proceeding."

  for FILE_DIR in `cat ${DIR_LIST_TXT}`; do
    for FILE in "${FILE_REMOVE_LIST[@]}"; do
      #echo "Removing ./${FILE_DIR}/${FILE}"
      rm "./${FILE_DIR}/${FILE}"
    done
  done
else
  echo "Aborting ..."
fi

