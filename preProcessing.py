def preProcess(arr):
    # Quantize pixel values to be multiples of 5 only
    for i in range(arr.size):
        diff = arr[i] % 5
        arr[i] = arr[i] - diff if 0 <= arr[i] % 5 <= 2 else arr[i] + 5 - diff

    print(arr)

# TODO: Eb2a 5alleehom talatat
