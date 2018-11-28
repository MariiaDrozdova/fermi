for file in *real_*map.fits
do
    echo "${file}"
    mv -i "${file}" ../database2/"${file}"
done
