def preProcess(arr):
    # Quantize pixel values to be multiples of 5 only
    for i in range(arr.size):
        if arr[i] == 0:
            arr[i] += 3
        elif arr[i] == 255:
            arr[i] -= 2
        elif arr[i] % 10 == 4 or 7 or 0:
            arr[i] -= 1
        elif arr[i] % 10 == 2 or 5 or 8:
            arr[i] += 1
        elif arr[i] % 10 == 1:
            arr[i] += 2
