noise_levels=(2.5 5 10 20 40 80)
USAGE="Usage: $0 -<flag/option> <filename>"

if [[ $# -lt  2 ]]; then
	echo $USAGE
	echo "-f -- archive list"
	echo "-s -- where to save"
	exit 1
fi

while (($#)); do
	case "$1" in
		-f) 
			shift
			models=$1
			shift;;
		-s)
			shift
			save_dir=$1
			shift;;
		*)
			echo "Wrong Option"
			echo $USAGE
			exit 1;;
	esac
done

if ! [[ -v models ]]; then
	echo "Undefineded Models"
	exit 1
fi

if ! [[ -v save_dir ]]; then
	save_dir="noise_results"
fi

for n in "${noise_levels[@]}"; do
	python noise_test_II.py -m $models -n $n -s $save_dir --gpu /GPU:1 --verbose 2>> noise_err.txt
done

exit 0
