#!/bin/bash
URL="https://github.com/kylebgorman/wikipron.git"
TMP_DESTINATION="data/phoneme/wikipron/temp"
DESTINATION="data/phoneme/wikipron"

echo Downloading
git clone --quiet $URL $TMP_DESTINATION
echo Finished downloading

# extract language codes from filenames
for code in $(ls $TMP_DESTINATION/data/wikipron/tsv/*_phonemic.tsv | sed 's/_[^_]*$//g' | cut -d / -f 8)
do
mkdir $DESTINATION/$code
cp $TMP_DESTINATION/data/wikipron/tsv/${code}_phonemic.tsv $DESTINATION/$code/raw.tsv
done
rm -rf $TMP_DESTINATION