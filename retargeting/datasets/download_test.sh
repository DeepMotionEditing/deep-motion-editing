export fileid=1_849LvuT3WBEHktBT97P2oMBzeJz7-UP
export filename=test_set.tar.bz2

wget -O $filename 'https://docs.google.com/uc?export=download&id='$fileid

tar -jxvf $filename
rm $filename
