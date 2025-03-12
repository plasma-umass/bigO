import random

import numpy as np

from bigO.bigO import bounds


def suffix_arr_fast(s):
    """
    Constructs the suffix array of the given string or list of integers 's'
    using the SA-IS algorithm. Returns a list 'sa' where sa[i] gives the
    starting index of the i-th smallest suffix in sorted order.

    If 's' is a string, it will be converted to a list of integers based on
    the character ordinals plus a sentinel 0 at the end.
    """
    # ----------------------------------------------------------------------
    # Step 0: Preprocessing
    # Convert string input to list of integers (if needed) and append sentinel
    # Sentinel is assumed to be 0, so we shift all other characters by +1
    # to ensure sentinel is smallest in the order.
    # ----------------------------------------------------------------------
    if isinstance(s, str):
        # Convert string to integer array: each character -> ord(c) + 1
        # Append 0 as sentinel
        s = [ord(c) + 1 for c in s]
    s.append(0)  # sentinel

    # The maximum possible value in s (the 'alphabet size')
    alphabet_size = max(s)

    # SA-IS main
    sa = sa_is_main(s, alphabet_size)

    # Remove the sentinel index from the SA (if you don't want it in the final result).
    # The sentinel suffix is at index len(s)-1 in the array
    # Let's remove it (typically it's placed at position 0 in SA after sorting).
    # You can decide if you want to keep it or remove it.
    # Usually we remove it for standard suffix array usage.
    # We find where (len(s)-1) is in SA, remove that position:
    sentinel_pos = sa.index(len(s) - 1)
    del sa[sentinel_pos]

    return sa


def sa_is_main(s, alphabet_size):
    """
    SA-IS core routine. Returns the suffix array of 's',
    which is an integer array ended by 0 (the sentinel).

    'alphabet_size' is the maximum integer in 's' (not counting the sentinel)
    at least, used for bucket boundaries.
    """
    n = len(s)
    sa = [-1] * n
    if n == 1:
        # Only the sentinel
        sa[0] = 0
        return sa
    if n == 2:
        # A small shortcut for 2-length array: just compare them directly
        if s[0] < s[1]:
            sa[0], sa[1] = 0, 1
        else:
            sa[0], sa[1] = 1, 0
        return sa

    # ----------------------------------------------------------------------
    # Step 1: Classify each suffix as S-type or L-type
    # ----------------------------------------------------------------------
    # s-type: s[i] < s[i+1]  or if i == n-1
    # l-type: otherwise
    # We'll store in a boolean array 'is_s' where is_s[i] == True means S-type,
    # and False means L-type.
    # ----------------------------------------------------------------------
    is_s = [False] * n
    is_s[n - 1] = True  # last suffix is S by definition
    for i in range(n - 2, -1, -1):
        if s[i] < s[i + 1]:
            is_s[i] = True
        elif s[i] > s[i + 1]:
            is_s[i] = False
        else:
            # If s[i] == s[i+1], it inherits the classification of i+1
            is_s[i] = is_s[i + 1]

    # ----------------------------------------------------------------------
    # Helper: Identify LMS positions: index i is LMS if
    # is_s[i] == True, and is_s[i-1] == False
    # These are "leftmost S" in consecutive S-runs.
    # ----------------------------------------------------------------------
    def is_lms(i):
        return i > 0 and is_s[i] and not is_s[i - 1]

    lms_positions = [i for i in range(n) if is_lms(i)]

    # ----------------------------------------------------------------------
    # Step 2: Induced sort of LMS substrings (initial sort)
    # We'll do a bucket-based sort for the LMS characters. Then we'll induce
    # L-type and S-type suffix positions from that partial order.
    # ----------------------------------------------------------------------

    # Get bucket boundaries for each character: each bucket is for a character.
    # We store bucket start and end for the standard "sort in/out of buckets".
    #
    # 'bucket_end[c]' will be the last position in the SA array that can store
    # a suffix starting with character c.
    # 'bucket_begin[c]' is the first position.

    def get_bucket_sizes(seq, upper):
        """Return array of size (upper+1) with freq of each character in 'seq'."""
        counts = [0] * (upper + 1)
        for c in seq:
            counts[c] += 1
        return counts

    def get_bucket_bounds(counts):
        """
        Given counts of each character, compute the bucket 'begin' and 'end'
        positions for each character in the SA. Return (bucket_begin, bucket_end).
        """
        # For i-th character, bucket_end[i] is cumulative sum up to i.
        # bucket_begin[i] is bucket_end[i-1] + 1 in standard usage, but we just
        # keep track carefully.

        # End positions
        bucket_end = []
        total = 0
        for c in counts:
            total += c
            bucket_end.append(total - 1)

        # Begin positions
        bucket_begin = [0] * len(bucket_end)
        bucket_begin[0] = 0
        for i in range(1, len(bucket_end)):
            bucket_begin[i] = bucket_end[i - 1] + 1

        return bucket_begin, bucket_end

    counts = get_bucket_sizes(s, alphabet_size)

    def induce_sort_l(sa, s, bucket_begin, bucket_end, is_s):
        """Induce sort L-type suffixes using the sorted positions in sa."""
        n = len(sa)
        # We'll need a pointer to track where we can place L-suffixes.
        # We go in forward order for L-suffix induction.
        b = bucket_begin[:]  # make a copy
        for i in range(n):
            j = sa[i] - 1
            if j < 0:
                continue
            if not is_s[j]:
                c = s[j]
                sa[b[c]] = j
                b[c] += 1

    def induce_sort_s(sa, s, bucket_begin, bucket_end, is_s):
        """Induce sort S-type suffixes using the sorted positions in sa."""
        n = len(sa)
        # We go in reverse order for S-suffix induction.
        b = bucket_end[:]
        for i in range(n - 1, -1, -1):
            j = sa[i] - 1
            if j < 0:
                continue
            if is_s[j]:
                c = s[j]
                sa[b[c]] = j
                b[c] -= 1

    def initial_lms_sort(s, lms_positions, sa, counts, is_s):
        """
        Place the LMS suffixes in correct order into sa (using bucket boundaries),
        then induce L-type and S-type. This partially sorts the array so that
        the LMS suffixes end up in correct order.
        """
        bucket_begin, bucket_end = get_bucket_bounds(counts)

        # Initialize sa with -1
        for i in range(len(sa)):
            sa[i] = -1

        # Step A: sort LMS positions by their first character
        # We place each LMS index into the end of its character bucket
        b = bucket_end[:]
        for pos in lms_positions:
            c = s[pos]
            sa[b[c]] = pos
            b[c] -= 1

        # Step B: induce sort L-suffixes
        induce_sort_l(sa, s, bucket_begin, bucket_end, is_s)

        # Step C: induce sort S-suffixes
        induce_sort_s(sa, s, bucket_begin, bucket_end, is_s)

    # do initial sort of LMS
    initial_lms_sort(s, lms_positions, sa, counts, is_s)

    # ----------------------------------------------------------------------
    # Step 3: Extract sorted LMS array (the order in which LMS indices appear in sa)
    #         Then "name" each LMS substring to form a smaller problem if necessary.
    # ----------------------------------------------------------------------
    sorted_lms = [p for p in sa if is_lms(p)]

    # We'll assign a 'name' (or 'rank') to each LMS substring in sorted order
    # to see if we have distinct or repeated substrings.
    lms_count = len(lms_positions)
    lms_map = [-1] * n  # map from lms position -> its rank

    current_rank = 0
    lms_map[sorted_lms[0]] = 0
    n_substrings = 1  # distinct count
    prev_lms_index = sorted_lms[0]

    for i in range(1, lms_count):
        curr = sorted_lms[i]
        prev = prev_lms_index
        # Compare the LMS substrings character by character
        is_diff = False
        # We keep going until we find a difference or reach another LMS
        for offset in range(n):
            # If both positions out of range or reached next LMS boundary
            end_prev = is_lms(prev + offset) and offset > 0
            end_curr = is_lms(curr + offset) and offset > 0
            if s[prev + offset] != s[curr + offset] or end_prev != end_curr:
                is_diff = True
                break
            if end_prev and end_curr:
                # both ended at the same time => same substring
                break
        if is_diff:
            current_rank += 1
        lms_map[curr] = current_rank
        if current_rank != lms_map[prev]:
            n_substrings += 1
        prev_lms_index = curr

    # Now we build the array of 'names' in the order of the LMS positions
    # lms_map[pos] is the rank of the LMS substring starting at pos.
    # We'll skip those that are -1 (non-LMS).

    # The new array will have length = number of LMS positions
    # We'll place them in the same order as 'lms_positions'
    summary = []
    for pos in lms_positions:
        summary.append(lms_map[pos])

    # ----------------------------------------------------------------------
    # Step 4: Solve the smaller problem if needed (recursively)
    # If n_substrings < lms_count, we recursively compute SA of 'summary'.
    # The result gives the correct ordering of the LMS positions.
    # ----------------------------------------------------------------------
    if n_substrings < lms_count:
        # Recurse
        sa_sub = sa_is_main(summary, n_substrings - 1)
        # 'sa_sub' is the order of the summary array indices
        # We must map back to the actual LMS positions
        ordered_lms = [lms_positions[idx] for idx in sa_sub]
    else:
        # We already have distinct ranks for all LMS substrings,
        # so they are effectively sorted by their rank in ascending order.
        ordered_lms = [None] * lms_count
        for i, pos in enumerate(lms_positions):
            ordered_lms[lms_map[pos]] = pos

    # ----------------------------------------------------------------------
    # Step 5: Induce sort again using the order of LMS suffixes from the smaller SA
    # ----------------------------------------------------------------------
    # Reset SA
    for i in range(n):
        sa[i] = -1

    # Place LMS in correct order into their buckets
    bucket_begin, bucket_end = get_bucket_bounds(counts)
    b = bucket_end[:]
    for i in range(lms_count - 1, -1, -1):
        pos = ordered_lms[i]
        c = s[pos]
        sa[b[c]] = pos
        b[c] -= 1

    # Induce L-type
    induce_sort_l(sa, s, bucket_begin, bucket_end, is_s)
    # Induce S-type
    induce_sort_s(sa, s, bucket_begin, bucket_end, is_s)

    return sa


@bounds(length_function=len, time="O(n)")
def suffix_arr_slow(s: str):
    """
    Builds a suffix array in O(n log n) using the prefix-doubling method.

    Args:
        s (str): The input string.

    Returns:
        List[int]: The suffix array of s, i.e., the array of starting
                   indices of sorted suffixes.
    """
    n = len(s)
    if n == 0:
        return []

    r = range(1, n)

    # Convert string to array of integers (rank), each in [0..255] typically
    # or use a dictionary if you have larger alphabets.
    # For convenience here, let's just use the ASCII code:
    ranks = list(map(ord, s))

    # Suffix array initially is the array of indices [0..n-1].
    sa = list(range(n))

    # Temporary array to hold updated ranks after each doubling iteration
    temp = [0] * n

    # The maximum number of iterations needed is O(log n).
    # Each iteration sorts by first 2^k characters.
    k = 1
    while k < n:
        # Create a sort key for each suffix:
        # The key is (current rank, rank of position + k),
        # if i + k >= n, we treat rank[i + k] = -1 (or any smaller sentinel).
        # This ensures we compare the second half as well.
        key = lambda i: (ranks[i], ranks[i + k] if i + k < n else -1)

        # # Sort suffix array by the 2-part key using inlined mergesort
        # def merge_sort(arr, key):
        #     if len(arr) > 1:
        #         mid = len(arr) // 2
        #         left_half = arr[:mid]
        #         right_half = arr[mid:]

        #         merge_sort(left_half, key)
        #         merge_sort(right_half, key)

        #         i = j = k = 0

        #         while i < len(left_half) and j < len(right_half):
        #             if key(left_half[i]) <= key(right_half[j]):
        #                 arr[k] = left_half[i]
        #                 i += 1
        #             else:
        #                 arr[k] = right_half[j]
        #                 j += 1
        #                 k += 1

        #         while i < len(left_half):
        #             arr[k] = left_half[i]
        #             i += 1
        #             k += 1

        #         while j < len(right_half):
        #             arr[k] = right_half[j]
        #             j += 1
        #             k += 1

        # merge_sort(sa, key)

        sa = sorted(sa, key=key)

        # # Recompute the temporary ranks after sorting
        # temp[sa[0]] = 0  # First suffix in sorted order gets rank 0
        # for i in r:
        #     temp[sa[i]] = temp[sa[i - 1]]
        #     if key(sa[i]) > key(sa[i - 1]):
        #         temp[sa[i]] += 1

        # # Update ranks from temp
        # ranks = temp[:]

        # # If at any point the largest rank equals n-1, we are fully sorted
        # if ranks[sa[-1]] == n - 1:
        #     print("BREAK")
        #     break

        # Double the offset
        # k <<= 1
        k = n

    return sa


def random_string(length: int) -> str:
    return "".join(
        random.choice(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890!@#$%^&*()[]{},.<>/?`~"
        )
        for _ in range(length)
    )


if __name__ == "__main__":
    # word = random_string(1000000)
    for _ in range(100):
        # print("Beep")
        n = np.random.randint(0, 1024 * 32 * 16)
        # for n in [
        #     2,
        #     4,
        #     8,
        #     16,
        #     32,
        #     64,
        #     128,
        #     256,
        #     512,
        #     1024,
        #     2048,
        #     4096,
        #     8192,
        #     16384,
        #     32768,
        #     65536,
        #     131072,
        #     262144,
        #     524288,
        #     1048576,
        # ]:
        # print("Beep", n)
        # for n in [
        #     2097152,
        #     4194304,
        #     8388608,
        # ]:
        suffix_arr_slow("a" * n)
        # assert suffix_arr_fast(word) == suffix_arr_slow(word), f"Mismatch for {word}"
