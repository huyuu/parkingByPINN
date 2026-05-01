def fun1(arr):
    n = arr[0]
    data = arr[1:]
    data.sort()

    result = []
    for i in range(1, len(data)):
        start_index = min(i - n + 1, 0)
        window = data[start_index: i + 1]
        window_len = len(window)

        if window_len % 2 == 0: # even
            median = (window[window_len//2 - 1] + window[window_len//2]) / 2
        else:
            median = window[window_len//2]

        if median == int(median):
            result.append(str(int(median)))
        else:
            result.append(median)
        result.append(median)
    return result



def fun2(strParam, num):
    result = []
    for ch in strParam:
        if ch.isalpha():
            base = ord('a') if ch.islower() else ord('A')
            shifted = (ord(ch) - base + num) % 26
            result.append(chr(shifted))
        else:
            result.append(ch)
    
    return ''.join(result)


def fun3(arr):
    n = arr[0]
    data = arr[1:]

    combined = data[n:] + data[:n]
    return ",".join(map(str, combined))


def fun4(arr):
    n = len(arr)
    total_sum = sum(arr)
    target_sum = total_sum // 2
    target_size = n // 2

    arr.sort()

    def solve(current_index, current_sum, current_path):
        if len(current_path) == target_size:
            if current_sum == target_sum:
                return current_path
            return None
        
        # early return if the remaining elements cannot reach the target size
        remaining_elements = n - current_index
        if remaining_elements + len(current_path) < target_size:
            return None

        for i in range(current_index, n):
            # if adding the next element exceeds the target sum
            if current_sum + arr[i] > target_sum:
                break

            result = solve(current_index + 1, current_sum + arr[current_index], current_path + [arr[current_index]])
            if result is not None:
                return result
            
    set1 = solve(0, 0, [])
    set2 = list(arr)

    



