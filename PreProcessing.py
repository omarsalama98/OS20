def preProcess(arr, quality):
    # Quantize pixel values to be multiples of 5 only
    if quality == 2:
        for i in range(arr.size):
            diff = arr[i] % 5
            arr[i] = arr[i] - diff if 0 < arr[i] % 10 <= 5 else arr[i] + 5 - diff
    elif quality == 3:
        for i in range(arr.size):
            if arr[i] == 0:
                arr[i] += 3
            elif arr[i] == 255:
                arr[i] -= 2
            elif arr[i] % 10 == 4 or arr[i] % 10 == 7 or arr[i] % 10 == 0:
                arr[i] -= 1
            elif arr[i] % 10 == 2 or arr[i] % 10 == 5 or arr[i] % 10 == 8:
                arr[i] += 1
            elif arr[i] % 10 == 1:
                arr[i] += 2
    elif quality == 4:
        for i in range(arr.size):
            if arr[i] == 255:
                arr[i] -= 2
            elif arr[i] % 3 == 1:
                arr[i] -= 1
            elif arr[i] % 3 == 2:
                arr[i] += 1
    else:
        for i in range(arr.size):
            if arr[i] == 255:
                arr[i] -= 1
            elif arr[i] % 2 != 0:
                arr[i] += 1
