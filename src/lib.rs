use std::cell::RefCell;
use std::rc::Rc;

// Add Digits
pub fn add_digits(num: i32) -> i32 {
    let mut tmp = num;
    loop {
        if tmp < 10 {
            return tmp;
        }
        let mut tmp2 = 0;
        loop {
            if tmp == 0 {
                break;
            }
            let m = tmp % 10;
            tmp2 += m;
            tmp /= 10;
        }
        tmp = tmp2;
    }
}

// Total Hamming Distance
pub fn total_hamming_distance(nums: Vec<i32>) -> i32 {
    let (mut i, mut res) = (0, 0);
    while i < nums.len() {
        let mut j = i + 1;
        while j < nums.len() {
            res += (nums[i] ^ nums[j]).count_ones() as i32;
            j += 1;
        }
        i += 1;
    }
    res
}

// Minimum Moves to Equal Array Elements II
pub fn min_moves2(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let m = nums[nums.len() / 2];
    nums.iter().map(|x| (x - m).abs()).sum()
}

// Repeated Substring Pattern
pub fn repeated_substring_pattern(s: String) -> bool {
    for i in 1..(s.len() / 2 + 1) {
        if s.len() % i != 0 {
            continue;
        }
        let subs = &s[..i];
        if s.eq(&subs.repeat(s.len() / i)) {
            return true;
        }
    }
    false
}

// Island Perimeter
pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            match grid[i][j] {
                1 => {
                    res += 2;
                    if i > 0 {
                        res -= grid[i - 1][j];
                    }
                    if j > 0 {
                        res -= grid[i][j - 1];
                    }
                }
                _ => (),
            }
        }
    }
    res * 2
}

// Target Sum
pub fn find_target_sum_ways(nums: Vec<i32>, s: i32) -> i32 {
    fn find(nums: &Vec<i32>, s: i32, i: usize, t: i32, r: &mut i32) {
        if i >= nums.len() {
            if t == s {
                *r += 1;
            }
            return;
        }
        find(nums, s, i + 1, t + nums[i], r);
        find(nums, s, i + 1, t + (-nums[i]), r);
    }
    let r = &mut 0i32;
    find(&nums, s, 0, 0, r);
    *r
}

// Find All Duplicates in an Array
pub fn find_duplicates(nums: Vec<i32>) -> Vec<i32> {
    let (mut m, mut res) = (std::collections::HashMap::new(), vec![]);

    for v in nums.iter() {
        if let Some(_) = m.get(v) {
            res.push(*v);
        } else {
            m.insert(v, 1);
        }
    }

    res
}

// Reverse Vowels of a String
pub fn reverse_vowels(s: String) -> String {
    fn is_vowel(c: char) -> bool {
        match c {
            'a' | 'e' | 'i' | 'o' | 'u' => true,
            _ => false,
        }
    }

    let (mut i, mut j) = (0, s.len().checked_sub(1).unwrap_or(0));
    let mut cs = s.chars().collect::<Vec<char>>();

    while i < j {
        if !is_vowel(cs[i].to_ascii_lowercase()) {
            i += 1;
            continue;
        }

        if !is_vowel(cs[j].to_ascii_lowercase()) {
            j -= 1;
            continue;
        }

        cs.swap(i, j);
        i += 1;
        j -= 1;
    }
    cs.iter().collect()
}

// Missing Number
pub fn missing_number(nums: Vec<i32>) -> i32 {
    let mut s1 = nums.len();
    let mut s2 = 0;
    for (i, v) in nums.iter().enumerate() {
        s1 += i;
        s2 += v;
    }
    s1 as i32 - s2
}

// Two Sum
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    use std::collections::HashMap;

    let mut m: HashMap<i32, i32> = HashMap::new();
    for (i, v) in nums.iter().enumerate() {
        match m.get(&(target - *v)) {
            Some(&i2) => return vec![i as i32, i2],
            None => m.insert(*v, i as i32),
        };
    }
    vec![]
}

// Definition for singly-linked list.
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct ListNode {
//   pub val: i32,
//   pub next: Option<Box<ListNode>>
// }
//
// impl ListNode {
//   #[inline]
//   fn new(val: i32) -> Self {
//     ListNode {
//       next: None,
//       val
//     }
//   }
// }
// Add Two Numbers
pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    match (l1, l2) {
        (None, None) => None,
        (Some(n), None) | (None, Some(n)) => Some(n),
        (Some(n1), Some(n2)) => {
            let sum = n1.val + n2.val;
            match sum {
                0..=9 => Some(Box::new(ListNode {
                    val: sum,
                    next: Solution::add_two_numbers(n1.next, n2.next),
                })),
                _ => Some(Box::new(ListNode {
                    val: sum - 10,
                    next: Solution::add_two_numbers(
                        Solution::add_two_numbers(Some(Box::new(ListNode::new(1))), n1.next),
                        n2.next,
                    ),
                })),
            }
        }
    }
}

// Reverse Integer
pub fn reverse(x: i32) -> i32 {
    x.signum()
        * x.abs()
            .to_string()
            .chars()
            .rev()
            .collect::<String>()
            .parse::<i32>()
            .unwrap_or(0)
}

// Longest Substring Without Repeating Characters
pub fn length_of_longest_substring(s: String) -> i32 {
    let mut map = std::collections::HashMap::new();
    let mut start = 0;
    let mut result = 0;
    for (i, c) in s.chars().enumerate() {
        map.entry(c)
            .and_modify(|x| {
                start = start.max(*x + 1);
                *x = i;
            })
            .or_insert(i);
        result = result.max(i - start + 1);
    }
    result as i32
}

// Median of Two Sorted Arrays
pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let mut merged = vec![];
    let (mut i, mut j) = (0, 0);

    while i < nums1.len() && j < nums2.len() {
        if nums1[i] < nums2[j] {
            merged.push(nums1[i]);
            i += 1;
        } else {
            merged.push(nums2[j]);
            j += 1;
        }
    }

    while i < nums1.len() {
        merged.push(nums1[i]);
        i += 1;
    }

    while j < nums2.len() {
        merged.push(nums2[j]);
        j += 1;
    }

    let mid = merged.len() / 2;
    match merged.len() % 2 {
        0 => return ((merged[mid - 1] as f64 + merged[mid] as f64) / 2.0) as f64,
        _ => return merged[mid] as f64,
    }
}

// Longest Valid Parentheses
pub fn longest_valid_parentheses(s: String) -> i32 {
    let mut stack = vec![];
    let mut m = 0;

    stack.push(-1);
    for (i, c) in s.chars().enumerate() {
        match c {
            '(' => stack.push(i as i32),
            _ => {
                stack.pop();
                match stack.is_empty() {
                    true => stack.push(i as i32),
                    false => {
                        let l = i as i32 - stack[stack.len() - 1];
                        m = m.max(l);
                    }
                }
            }
        }
    }
    m
}

// Longest Common Prefix
pub fn longest_common_prefix(strs: Vec<String>) -> String {
    match strs.is_empty() {
        true => "".to_string(),
        _ => strs.iter().skip(1).fold(strs[0].clone(), |acc, x| {
            acc.chars()
                .zip(x.chars())
                .take_while(|(x, y)| x == y)
                .map(|(x, _)| x)
                .collect()
        }),
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Binary Tree Inorder Traversal
pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let (mut res, mut stack) = (vec![], vec![]);
    let mut r = root.clone();

    while r.is_some() || !stack.is_empty() {
        while let Some(node) = r {
            stack.push(node.clone());
            r = node.borrow().left.clone();
        }
        r = stack.pop();
        if let Some(node) = r {
            res.push(node.borrow().val);
            r = node.borrow().right.clone();
        }
    }
    res
}

// Merge Sorted Array
pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    let (mut m, mut n) = (m as usize, n as usize);
    while m >= 1 && n >= 1 {
        if nums1[m - 1] > nums2[n - 1] {
            nums1[m + n - 1] = nums1[m - 1];
            m -= 1;
        } else {
            nums1[m + n - 1] = nums2[n - 1];
            n -= 1;
        }
    }

    while n >= 1 {
        nums1[m + n - 1] = nums2[n - 1];
        n -= 1;
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Same Tree
pub fn is_same_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
    match (p, q) {
        (None, None) => true,
        (Some(p), Some(q)) => {
            let p = p.borrow();
            let q = q.borrow();
            p.val == q.val
                && Self::is_same_tree(p.left.clone(), q.left.clone())
                && Self::is_same_tree(p.right.clone(), q.right.clone())
        }
        _ => false,
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Binary Tree Level Order Traversal
pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut res: Vec<Vec<i32>> = Vec::new();

    fn traversal(root: Option<Rc<RefCell<TreeNode>>>, res: &mut Vec<Vec<i32>>, level: usize) {
        if let Some(r) = root {
            if res.len() == level {
                res.push(Vec::new());
            }
            res[level].push(r.borrow().val);
            traversal(r.borrow().left.clone(), res, level + 1);
            traversal(r.borrow().right.clone(), res, level + 1);
        }
    }
    traversal(root.clone(), &mut res, 0);

    res
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Binary Tree Preorder Traversal
pub fn preorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    match root {
        Some(r) => {
            let (mut res, mut s) = (vec![], vec![r]);
            while let Some(p) = s.pop() {
                res.push(p.borrow().val);
                if let Some(right) = p.borrow().right.clone() {
                    s.push(right);
                }
                if let Some(left) = p.borrow().left.clone() {
                    s.push(left);
                }
            }
            res
        }
        None => vec![],
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
//
// Binary Tree Postorder Traversal
pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    match root {
        None => vec![],
        Some(root) => {
            let (mut s1, mut s2, mut res) = (vec![root], vec![], vec![]);

            while let Some(n) = s1.pop() {
                if let Some(left) = n.borrow().left.clone() {
                    s1.push(left);
                }
                if let Some(right) = n.borrow().right.clone() {
                    s1.push(right);
                }
                s2.push(n);
            }

            while let Some(n) = s2.pop() {
                res.push(n.borrow().val);
            }
            res
        }
    }
}

// Edit Distance
pub fn min_distance(word1: String, word2: String) -> i32 {
    let (l1, l2) = (word1.len(), word2.len());
    let (w1, w2) = (word1.into_bytes(), word2.into_bytes());

    let mut dp = vec![vec![0; l2 + 1]; l1 + 1];

    for i in 0..=l1 {
        dp[i][0] = i as i32;
    }

    for j in 0..=l2 {
        dp[0][j] = j as i32;
    }

    for i in 1..=l1 {
        for j in 1..=l2 {
            if w1[i - 1] == w2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]) + 1;
            }
        }
    }
    dp[l1][l2]
}

// Remove Duplicates from Sorted Array
pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    match nums.is_empty() {
        true => 0,
        false => {
            let mut prev = 0;
            for i in 1..nums.len() {
                if nums[prev] != nums[i] {
                    prev += 1;
                    nums[prev] = nums[i];
                }
            }
            (prev + 1) as i32
        }
    }
}

// Pascal's Triangle
pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
    let mut res = vec![];
    for i in 0..num_rows as usize {
        res.push(vec![1; i + 1]);
        for j in 1..i {
            res[i][j] = res[i - 1][j - 1] + res[i - 1][j];
        }
    }
    res
}

// Pascal's Triangle II
pub fn get_row(row_index: i32) -> Vec<i32> {
    let mut res: Vec<Vec<i32>> = vec![];
    for i in 0..=row_index as usize {
        let mut row: Vec<i32> = vec![1; i + 1];
        for j in 1..i {
            row[j] = res[i - 1][j - 1] + res[i - 1][j];
        }
        if i == row_index as usize {
            return row;
        }
        res.push(row);
    }
    vec![]
}

// Word Break
pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
    let mut v = vec![false; s.len() + 1];
    v[0] = true;
    for i in 0..v.len() {
        match v[i] {
            false => continue,
            true => {
                for word in word_dict.iter() {
                    if let Some(s2) = s.get(i..i + word.len()) {
                        if s2 == *word {
                            v[i + word.len()] = true;
                        }
                    }
                }
            }
        }
    }
    v[s.len()]
}

// Count Primes
pub fn count_primes(n: i32) -> i32 {
    let mut prime = vec![true; n as usize];
    let mut res = 0;

    for i in 2..n as usize {
        if !prime[i] {
            continue;
        }

        res += 1;
        let mut mul = i * i;
        while mul < n as usize {
            prime[mul] = false;
            mul += i;
        }
    }

    res
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Kth Smallest Element in a BST
pub fn kth_smallest(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i32 {
    fn traversal(n: Option<Rc<RefCell<TreeNode>>>, tmp: &mut Vec<i32>) {
        match n {
            Some(n) => {
                traversal(n.borrow().left.clone(), tmp);
                // Kth Smallest Element in a BST
                tmp.push(n.borrow().val);
                traversal(n.borrow().right.clone(), tmp);
            }
            None => (),
        }
    }

    let mut tmp = vec![];
    traversal(root.clone(), &mut tmp);
    tmp[k as usize - 1]
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Count Complete Tree Nodes
pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn countn(n: Option<Rc<RefCell<TreeNode>>>, c: &mut i32) {
        match n {
            Some(n) => {
                *c += 1;
                countn(n.borrow().left.clone(), c);
                countn(n.borrow().right.clone(), c);
            }
            _ => (),
        }
    }
    let res = &mut 0i32;
    countn(root.clone(), res);
    *res
}

// Kth Largest Element in an Array
pub fn find_kth_largest(nums: Vec<i32>, k: i32) -> i32 {
    fn qSort(nums: &mut [i32]) {
        if nums.len() <= 1 {
            return;
        }

        let mid = nums.len() / 2;
        let (mut left, right) = (0, nums.len() - 1);
        nums.swap(mid, right);

        let mut i = 0;
        while i < nums.len() {
            if nums[i] < nums[right] {
                nums.swap(i, left);
                left += 1;
            }
            i += 1;
        }

        nums.swap(left, right);
        qSort(&mut nums[0..left]);
        qSort(&mut nums[left + 1..=right]);
    }

    let mut nums = nums;
    qSort(&mut nums);
    nums[nums.len() - k as usize]
}

// Combination Sum III
pub fn combination_sum3(k: i32, n: i32) -> Vec<Vec<i32>> {
    fn comb(k: usize, n: i32, mut cur: i32, l: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        if l.len() == k {
            return;
        }
        while cur <= 9 {
            let mut c = l.clone();
            let s = c.iter().sum::<i32>() + cur;

            if n < s {
                break;
            }

            c.push(cur);
            if s == n && c.len() == k {
                res.push(c.clone());
                break;
            }

            comb(k, n, cur + 1, &mut c, res);
            cur += 1;
        }
    }

    let (mut l, mut res) = (vec![], vec![]);
    comb(k as usize, n, 1, &mut l, &mut res);
    res
}

// Nim Game
pub fn can_win_nim(n: i32) -> bool {
    n & 3 != 0
}

// Power Of Three
pub fn is_power_of_three(n: i32) -> bool {
    let mut n = n;
    if n == 0 {
        false
    } else {
        while n % 3 == 0 {
            n /= 3
        }
        n == 1
    }
}

// Power of Four
pub fn is_power_of_four(num: i32) -> bool {
    match num <= 0 {
        true => false,
        _ => {
            let mut num = num;
            while num % 4 == 0 {
                num /= 4;
            }
            num == 1
        }
    }
}

// Sum of Two Integers
pub fn get_sum(a: i32, b: i32) -> i32 {
    fn sum(a: i32, b: i32) -> i32 {
        match b {
            0 => a,
            _ => sum(a ^ b, (a & b) << 1),
        }
    }
    sum(a, b)
}

// Find All Duplicates in an Array
pub fn find_duplicates(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    let mut rtn = vec![];

    for i in 0..nums.len() {
        let n = nums[i].abs();
        let idx = (n - 1) as usize;
        if nums[idx] < 0 {
            rtn.push(n);
        } else {
            nums[idx] = -1 * nums[idx];
        }
    }
    return rtn;
}

// Hamming Distance
pub fn hamming_distance(x: i32, y: i32) -> i32 {
    (x ^ y).count_ones() as i32
}

// Sort Characters By Frequency
pub fn frequency_sort(s: String) -> String {
    let mut counter = std::collections::HashMap::new();
    for c in s.chars() {
        let count = counter.entry(c).or_insert(0);
        *count += 1;
    }
    let mut list = counter.iter().collect::<Vec<(&char, &i32)>>();
    list.sort_unstable_by(|a, b| b.1.cmp(a.1));

    let mut res = vec![];
    for (&ch, &count) in list {
        for _ in 0..count {
            res.push(ch);
        }
    }
    res.iter().collect::<String>()
}

// Minimum Moves to Equal Array Elements
pub fn min_moves(nums: Vec<i32>) -> i32 {
    match nums.iter().min() {
        Some(m) => {
            let mut res = 0;
            for n in &nums {
                res += n - m;
            }
            res
        }
        _ => 0,
    }
}

// Ugly Number
pub fn is_ugly(num: i32) -> bool {
    if num < 1 {
        return false;
    }

    let facts = [2, 3, 5];
    let mut num = num;
    for v in facts.iter() {
        while num % v == 0 {
            num /= v;
        }
    }
    num == 1
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Find Mode in Binary Search Tree
pub fn find_mode(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    match root {
        None => vec![],
        Some(root) => {
            let mut counter = std::collections::HashMap::<i32, i32>::new();
            fn find(n: Option<Rc<RefCell<TreeNode>>>, m: &mut std::collections::HashMap<i32, i32>) {
                match n {
                    Some(n) => {
                        (*m.entry(n.borrow().val).or_insert(0)) += 1;
                        find(n.borrow().left.clone(), m);
                        find(n.borrow().right.clone(), m);
                    }
                    None => (),
                }
            }
            find(Some(root), &mut counter);
            let mut list = counter.iter().collect::<Vec<(&i32, &i32)>>();
            list.sort_unstable_by(|a, b| b.1.cmp(a.1));

            let (mut res, mut prev) = (vec![*list[0].0], list[0].1);
            for i in 1..list.len() {
                if prev != list[i].1 {
                    break;
                }
                res.push(*(list[i].0));
            }
            res
        }
    }
}

// Base 7
pub fn convert_to_base7(num: i32) -> String {
    let mut num = num;
    match num {
        0 => "0".to_string(),
        _ => {
            let sign = if num < 0 {
                num *= -1;
                "-".to_string()
            } else {
                "".to_string()
            };

            let mut s = String::new();
            while num > 0 {
                s = (num % 7).to_string() + &s;
                num /= 7;
            }
            sign + &s
        }
    }
}

// Relative Ranks
pub fn find_relative_ranks(nums: Vec<i32>) -> Vec<String> {
    let (mut ns, mut m) = (nums.clone(), std::collections::HashMap::new());
    ns.sort_by_key(|&n| -n);
    ns.into_iter().enumerate().for_each(|(i, n)| {
        m.insert(n, i);
    });

    nums.iter()
        .map(|n| match m[n] {
            0 => "Gold Medal".to_string(),
            1 => "Silver Medal".to_string(),
            2 => "Bronze Medal".to_string(),
            _ => (m[n] + 1).to_string(),
        })
        .collect()
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Find Bottom Left Tree Value
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn find_bottom_left_value(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn find(root: Option<Rc<RefCell<TreeNode>>>, height: i32, max: &mut i32, res: &mut i32) {
            match root {
                Some(root) => {
                    if height > *max {
                        *max = height;
                        *res = root.borrow().val;
                    }
                    find(root.borrow().left.clone(), height + 1, max, res);
                    find(root.borrow().right.clone(), height + 1, max, res);
                }
                _ => (),
            }
        }
        let (max, res) = (&mut 0, &mut 0);
        find(root.clone(), 1, max, res);
        *res
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Most Frequent Subtree Sum
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn find_frequent_tree_sum(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        match root {
            None => vec![],
            Some(root) => {
                let mut m = std::collections::HashMap::new();
                fn count(
                    root: Option<Rc<RefCell<TreeNode>>>,
                    m: &mut std::collections::HashMap<i32, i32>,
                ) -> i32 {
                    match root {
                        Some(root) => {
                            let left = count(root.borrow().left.clone(), m);
                            let right = count(root.borrow().right.clone(), m);
                            let total = left + right + root.borrow().val;
                            (*m.entry(total).or_insert(0)) += 1;
                            total
                        }
                        None => 0,
                    }
                }
                count(Some(root), &mut m);
                let mut list = m.iter().collect::<Vec<(&i32, &i32)>>();
                list.sort_unstable_by(|a, b| b.1.cmp(a.1));

                let (mut res, mut prev) = (vec![*list[0].0], list[0].1);
                for i in 1..list.len() {
                    if prev != list[i].1 {
                        break;
                    }
                    res.push(*list[i].0);
                }
                res

                /*
                or

                m.iter().fold(vec![], |mut acc, (&k, &v)| {
                    if v == max {
                        acc.push(v);
                    }
                    acc
                })
                */
            }
        }
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Find Largest Value in Each Tree Row
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn largest_values(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        fn collect(root: Option<Rc<RefCell<TreeNode>>>, level: usize, ret: &mut Vec<i32>) {
            match root {
                None => (),
                Some(root) => {
                    if ret.len() < level {
                        ret.push(root.borrow().val);
                    } else {
                        ret[level - 1] = ret[level - 1].max(root.borrow().val);
                    }
                    collect(root.borrow().left.clone(), level + 1, ret);
                    collect(root.borrow().right.clone(), level + 1, ret);
                }
            }
        }
        let mut ret = vec![];
        collect(root, 1, &mut ret);
        ret
    }
}

// Remove Element
impl Solution {
    pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
        nums.retain(|&v| v != val);
        nums.len() as i32
    }
}

// Find First and Last Position of Element in Sorted Array
// linear search
impl Solution {
    pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
        match nums.len() {
            0 => vec![-1, -1],
            _ => {
                let (mut start, mut end) = (-1, -1);
                for i in 0..nums.len() {
                    if nums[i] != target {
                        continue;
                    }

                    if start == -1 {
                        start = i as i32;
                        end = i as i32;
                    } else {
                        end = i as i32;
                    }
                }
                vec![start, end]
            }
        }
    }
}
//binary search
impl Solution {
    pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
        match nums.len() {
            0 => vec![-1, -1],
            _ => {
                let (mut left, mut right) = (0, nums.len() - 1);
                while left <= right && right < nums.len() {
                    let mid = (left + right) / 2;
                    if nums[mid] > target {
                        right = mid.checked_sub(1).unwrap_or(nums.len());
                    } else if nums[mid] < target {
                        left = mid + 1;
                    } else {
                        if nums[left] < target {
                            left += 1;
                        }
                        if nums[right] > target {
                            right -= 1;
                        }
                        if nums[left] == nums[right] {
                            return vec![left as i32, right as i32];
                        }
                    }
                }
                vec![-1, -1]
            }
        }
    }
}

// Search Insert Position
impl Solution {
    pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
        for (i, v) in nums.iter().enumerate() {
            if target <= *v {
                return i as i32;
            }
        }
        nums.len() as i32
    }
}

// Generate Parentheses
impl Solution {
    pub fn generate_parenthesis(n: i32) -> Vec<String> {
        fn back_track(s: String, open: i32, close: i32) -> Vec<String> {
            let mut res = vec![];
            if open == 0 && close == 0 {
                return vec![s];
            }
            if open > 0 {
                res.append(&mut back_track(s.clone() + "(", open - 1, close + 1));
            }
            if close > 0 {
                res.append(&mut back_track(s.clone() + ")", open, close - 1));
            }
            res
        }
        back_track("".to_string(), n, 0)
    }
}

//
// Definition for singly-linked list.
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct ListNode {
//   pub val: i32,
//   pub next: Option<Box<ListNode>>
// }
//
// impl ListNode {
//   #[inline]
//   fn new(val: i32) -> Self {
//     ListNode {
//       next: None,
//       val
//     }
//   }
// }
// Merge Two Sorted Lists
impl Solution {
    pub fn merge_two_lists(
        l1: Option<Box<ListNode>>,
        l2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        match (l1, l2) {
            (None, None) => None,
            (Some(n), None) | (None, Some(n)) => Some(n),
            (Some(l1), Some(l2)) => {
                if l1.val >= l2.val {
                    Some(Box::new(ListNode {
                        val: l2.val,
                        next: Solution::merge_two_lists(Some(l1), l2.next),
                    }))
                } else {
                    Some(Box::new(ListNode {
                        val: l1.val,
                        next: Solution::merge_two_lists(l1.next, Some(l2)),
                    }))
                }
            }
        }
    }
}

// Definition for singly-linked list.
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct ListNode {
//   pub val: i32,
//   pub next: Option<Box<ListNode>>
// }
//
// impl ListNode {
//   #[inline]
//   fn new(val: i32) -> Self {
//     ListNode {
//       next: None,
//       val
//     }
//   }
// }
// Swap Nodes in Pairs
impl Solution {
    pub fn swap_pairs(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        head.and_then(|mut n| match n.next {
            None => Some(n),
            Some(mut nn) => {
                n.next = Solution::swap_pairs(nn.next);
                nn.next = Some(n);
                Some(nn)
            }
        })
    }
}

// Pow(x, n)
impl Solution {
    pub fn my_pow(x: f64, n: i32) -> f64 {
        fn pow(x: f64, res: f64, n: i64) -> f64 {
            match n {
                0 => res,
                n if n & 1 == 1 => pow(x * x, res * x, n >> 1),
                _ => pow(x * x, res, n >> 1),
            }
        }
        match n {
            0 => 1.0,
            n if n < 0 => pow(1.0 / x, 1.0, (n as i64).abs()),
            _ => pow(x, 1.0, n as i64),
        }
    }
}

// Maximum Subarray
impl Solution {
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let (mut cur, mut max) = (nums[0], nums[0]);
        for i in 1..nums.len() {
            cur = match cur < 0 {
                true => nums[i],
                _ => cur + nums[i],
            };
            max = max.max(cur);
        }
        max
    }
}

// Remove Duplicates from Sorted Array II
impl Solution {
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        match nums.len() {
            0 | 1 => nums.len() as i32,
            _ => {
                let mut ptr = 2;
                for i in ptr..nums.len() {
                    if nums[ptr - 2] != nums[i] {
                        nums[ptr] = nums[i];
                        ptr += 1;
                    }
                }
                ptr as i32
            }
        }
    }
}
/*
// Below is a very cool solution
// https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/discuss/742918/Rust-cheapest-and-best
impl Solution {
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        match nums.len() {
            0 | 1 => nums.len() as i32,
            _ => (2..nums.len()).fold(2i32, |mut k, i| {
                if nums[(k - 2) as usize] != nums[i] {
                    nums[k as usize] = nums[i];
                    k += 1;
                }
                k
            }),
        }
    }
}
*/

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Recover Binary Search Tree
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn recover_tree(root: &mut Option<Rc<RefCell<TreeNode>>>) {
        fn dfs(
            node: &Option<Rc<RefCell<TreeNode>>>,
            first: &mut Option<Rc<RefCell<TreeNode>>>,
            second: &mut Option<Rc<RefCell<TreeNode>>>,
            prev: &mut Option<Rc<RefCell<TreeNode>>>,
        ) {
            if let Some(n) = node {
                dfs(&n.borrow().left, first, second, prev);
                if let Some(prev) = prev {
                    if prev.borrow().val >= n.borrow().val {
                        if first.is_none() {
                            *first = Some(prev.clone());
                        }
                        if first.is_some() {
                            *second = Some(n.clone());
                        }
                    }
                }
                *prev = Some(n.clone());
                dfs(&n.borrow().right, first, second, prev);
            }
        }
        let (mut first, mut second, mut prev) = (None, None, None);
        dfs(root, &mut first, &mut second, &mut prev);
        std::mem::swap(
            &mut first.unwrap().borrow_mut().val,
            &mut second.unwrap().borrow_mut().val,
        );
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Balanced Binary Tree
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn is_balanced(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn dfs(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
            match root {
                Some(root) => {
                    let left = dfs(root.borrow().left.clone());
                    let right = dfs(root.borrow().right.clone());
                    if (left - right).abs() > 1 || left == -1 || right == -1 {
                        return -1;
                    }
                    left.max(right) + 1
                }
                None => 0,
            }
        }
        dfs(root) != -1
    }
}

// Single Number
impl Solution {
    pub fn single_number(nums: Vec<i32>) -> i32 {
        nums.iter().fold(0, |acc, x| acc ^ x)
    }
}

// Single Number II
impl Solution {
    pub fn single_number(nums: Vec<i32>) -> i32 {
        fn q_sort(nums: &mut [i32]) {
            if nums.len() <= 1 {
                return;
            }

            let mid = nums.len() / 2;
            let (mut left, mut right) = (0, nums.len() - 1);
            nums.swap(mid, right);

            for i in 0..nums.len() {
                if nums[i] > nums[right] {
                    nums.swap(left, i);
                    left += 1;
                }
            }

            nums.swap(left, right);
            q_sort(&mut nums[0..left]);
            q_sort(&mut nums[left + 1..=right]);
        }

        let mut nums = nums;
        q_sort(&mut nums);

        let (mut res, mut i) = (0, 0);
        while i < nums.len() {
            if i + 1 == nums.len() || nums[i] != nums[i + 1] {
                return nums[i];
            }
            i += 3;
        }
        res
    }
}

// N-Repeated Element in Size 2N Array
impl Solution {
    pub fn repeated_n_times(a: Vec<i32>) -> i32 {
        let mut m = std::collections::HashSet::new();
        for x in a {
            if m.contains(&x) {
                return x;
            }
            m.insert(x);
        }
        unreachable!()
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Univalued Binary Tree
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn is_unival_tree(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn is_unival(n: Option<Rc<RefCell<TreeNode>>>, v: i32) -> bool {
            match n {
                None => true,
                Some(n) => {
                    n.borrow().val == v
                        && is_unival(n.borrow().left.clone(), v)
                        && is_unival(n.borrow().right.clone(), v)
                }
            }
        }
        is_unival(root.clone(), root.unwrap().borrow().val)
    }
}

// Reverse Only Letters
impl Solution {
    pub fn reverse_only_letters(s: String) -> String {
        let mut s: Vec<char> = s.chars().collect();
        let (mut start, mut end) = (0, s.len() - 1);
        loop {
            while start < s.len() {
                if s[start].is_alphabetic() {
                    break;
                }
                start += 1;
            }
            // checking if underflow
            while end != std::usize::MAX {
                if s[end].is_alphabetic() {
                    break;
                }
                end -= 1;
            }
            if start >= s.len() || end == std::usize::MAX || start >= end {
                break;
            }
            s.swap(start, end);
            start += 1;
            end -= 1;
        }
        s.iter().collect::<String>()
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Leaf-Similar Trees
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn leaf_similar(
        root1: Option<Rc<RefCell<TreeNode>>>,
        root2: Option<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        fn collect_leaves(n: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
            match n {
                None => vec![],
                Some(n) => {
                    if n.borrow().left.is_none() && n.borrow().right.is_none() {
                        return vec![n.borrow().val];
                    }
                    let mut list = collect_leaves(n.borrow().left.clone());
                    list.extend(collect_leaves(n.borrow().right.clone()));
                    list
                }
            }
        }
        collect_leaves(root1) == collect_leaves(root2)
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Maximum Level Sum of a Binary Tree
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn max_level_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn collect_levels(n: Option<Rc<RefCell<TreeNode>>>, level: usize, list: &mut Vec<i32>) {
            match n {
                None => (),
                Some(n) => {
                    if level == list.len() + 1 {
                        list.push(n.borrow().val);
                    } else {
                        list[level - 1] += n.borrow().val;
                    }
                    collect_levels(n.borrow().left.clone(), level + 1, list);
                    collect_levels(n.borrow().right.clone(), level + 1, list);
                }
            }
        }
        let mut list = vec![];
        collect_levels(root, 1, &mut list);
        let (mut max_index, mut max_value) = (0, std::i32::MIN);

        for i in 0..list.len() {
            if list[i] > max_value {
                max_value = list[i];
                max_index = i + 1;
            }
        }
        max_index as i32
    }
}

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
// Validate Binary Search Tree
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn validate(
            n: Option<Rc<RefCell<TreeNode>>>,
            min: Option<Rc<RefCell<TreeNode>>>,
            max: Option<Rc<RefCell<TreeNode>>>,
        ) -> bool {
            match n {
                None => true,
                Some(n) => {
                    if let Some(max) = max.clone() {
                        if max.borrow().val <= n.borrow().val {
                            return false;
                        }
                    }
                    if let Some(min) = min.clone() {
                        if min.borrow().val >= n.borrow().val {
                            return false;
                        }
                    }
                    validate(n.borrow().left.clone(), min, Some(n.clone()))
                        && validate(n.borrow().right.clone(), Some(n.clone()), max)
                }
            }
        }
        validate(root, None, None)
    }
}

// LRU Cache
use std::collections::{HashMap, VecDeque};

struct LRUCache {
    q: VecDeque<i32>,
    m: HashMap<i32, i32>,
    c: usize,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl LRUCache {
    fn new(capacity: i32) -> Self {
        Self {
            q: VecDeque::with_capacity(capacity as usize),
            m: HashMap::new(),
            c: capacity as usize,
        }
    }

    fn get(&mut self, key: i32) -> i32 {
        match self.m.get(&key) {
            Some(v) => {
                self.q
                    .remove(self.q.iter().position(|&x| x == key).unwrap());
                self.q.push_front(key);
                *v
            }
            None => -1,
        }
    }

    fn put(&mut self, key: i32, value: i32) {
        match self.m.get(&key) {
            Some(_) => {
                self.q
                    .remove(self.q.iter().position(|&x| x == key).unwrap());
                self.q.push_front(key);
                self.m.insert(key, value);
            }
            None => {
                if self.m.len() == self.c {
                    self.m.remove(&self.q.pop_back().unwrap());
                }
                self.m.insert(key, value);
                self.q.push_front(key);
            }
        }
    }
}

// Number of 1 Bits
impl Solution {
    pub fn hammingWeight(n: u32) -> i32 {
        let (mut n, mut count) = (n, 0);
        while n > 0 {
            if n & 1 == 1 {
                count += 1;
            }
            n >>= 1;
        }
        count
    }
}

// Reverse Bits
impl Solution {
    pub fn reverse_bits(x: u32) -> u32 {
        let (mut res, mut x) = (0u32, x);
        for _ in 0..32 {
            res = (res << 1) | (x & 1);
            x >>= 1;
        }
        res
    }
}

// Search in Rotated Sorted Array
impl Solution {
    pub fn search(nums: Vec<i32>, target: i32) -> i32 {
        let (mut left, mut right) = (0, nums.len() - 1);
        while left <= right && right != std::usize::MAX {
            if nums[left] == target {
                return left as i32;
            } else if nums[right] == target {
                return right as i32;
            }

            let mid = (right + left) / 2;
            if nums[mid] == target {
                return mid as i32;
            }

            if nums[left] > nums[right] {
                left += 1;
                right -= 1;
            } else {
                if nums[mid] > target {
                    right = mid - 1;
                    left += 1;
                } else {
                    left = mid + 1;
                    right -= 1;
                }
            }
        }
        -1
    }
}

// Longest Palindromic Substring
impl Solution {
    pub fn longest_palindrome(s: String) -> String {
        let (mut s, mut max) = (s.chars().collect::<Vec<char>>(), vec![]);
        fn find_max(s: &Vec<char>, max: Vec<char>, i: usize, j: usize) -> Vec<char> {
            let (mut i, mut j) = (i, j);
            let mut sub: &[char] = &[];
            while i != std::usize::MAX && j < s.len() && s[i] == s[j] {
                sub = &s[i..j + 1];
                i -= 1;
                j += 1;
            }
            if sub.len() > max.len() {
                return sub.to_vec();
            }
            max.to_vec()
        }
        for i in 0..s.len() {
            max = find_max(&s, max, i, i);
            max = find_max(&s, max, i, i + 1);
        }
        max.into_iter().collect()
    }
}

// String to Integer (atoi)
impl Solution {
    pub fn my_atoi(s: String) -> i32 {
        let (mut start, mut res, mut sign) = (false, 0i64, 1);

        for c in s.chars() {
            match c {
                '0'..='9' => {
                    start = true;
                    res = res * 10 + (c as i64 - '0' as i64);
                    if res > std::i32::MAX as i64 {
                        break;
                    }
                }
                ' ' => {
                    if start {
                        break;
                    }
                }
                '+' => {
                    if start {
                        break;
                    }
                    sign = 1;
                    start = true;
                }
                '-' => {
                    if start {
                        break;
                    }
                    sign = -1;
                    start = true;
                }
                _ => break,
            }
        }

        res *= sign;
        if res < std::i32::MIN as i64 {
            return std::i32::MIN;
        } else if res > std::i32::MAX as i64 {
            return std::i32::MAX;
        }
        res as i32
    }
}

// Valid Parentheses
impl Solution {
    pub fn is_valid(s: String) -> bool {
        let mut stack = vec![];
        for c in s.chars() {
            match c {
                ')' => match stack.pop() {
                    Some(c) => {
                        if c != '(' {
                            return false;
                        }
                    }
                    None => return false,
                },
                ']' => match stack.pop() {
                    Some(c) => {
                        if c != '[' {
                            return false;
                        }
                    }
                    None => return false,
                },
                '}' => match stack.pop() {
                    Some(c) => {
                        if c != '{' {
                            return false;
                        }
                    }
                    None => return false,
                },
                _ => stack.push(c),
            }
        }
        stack.is_empty()
    }
}

// Integer to Roman
impl Solution {
    pub fn int_to_roman(num: i32) -> String {
        let m = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
        let s = [
            "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I",
        ];

        let (mut num, mut buf) = (num, vec![]);
        for i in 0..13 {
            let mut j = num / m[i];
            num %= m[i];
            while j > 0 {
                buf.push(s[i]);
                j -= 1;
            }
        }
        buf.into_iter().collect()
    }
}

// Roman to Integer
impl Solution {
    pub fn roman_to_int(s: String) -> i32 {
        let t: std::collections::HashMap<char, i32> = [
            ('I', 1),
            ('V', 5),
            ('X', 10),
            ('L', 50),
            ('C', 100),
            ('D', 500),
            ('M', 1000),
        ]
        .iter()
        .cloned()
        .collect();

        let mut cs: Vec<char> = s.chars().collect();
        let mut res = *t.get(&cs[cs.len() - 1]).unwrap();
        let mut i = cs.len() - 2;

        while i != std::usize::MAX {
            let (current, next) = (t.get(&cs[i]).unwrap(), t.get(&cs[i + 1]).unwrap());
            if current < next {
                res -= *current;
            } else {
                res += *current;
            }
            i -= 1;
        }

        res
    }
}

// 3Sum
impl Solution {
    pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut nums = nums;
        nums.sort();
        let mut res = vec![];

        for i in 0..nums.len() {
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }
            let mut left = i + 1;
            let mut right = nums.len() - 1;

            while left < right {
                if left > i + 1 && nums[left] == nums[left - 1] {
                    left += 1;
                    continue;
                }
                if right < nums.len() - 1 && nums[right] == nums[right + 1] {
                    right -= 1;
                    continue;
                }

                let sum = nums[i] + nums[left] + nums[right];
                if sum == 0 {
                    res.push([nums[i], nums[left], nums[right]].to_vec());
                    left += 1;
                } else if sum < 0 {
                    left += 1;
                } else {
                    right -= 1;
                }
            }
        }

        res
    }
}

// 3Sum Closest
impl Solution {
    pub fn three_sum_closest(nums: Vec<i32>, target: i32) -> i32 {
        let mut nums = nums;
        nums.sort();
        let mut res: i64 = std::i32::MAX as i64;
        let target: i64 = target as i64;

        for i in 0..nums.len() - 2 {
            let mut left = i + 1;
            let mut right = nums.len() - 1;

            while left < right {
                let sum: i64 = (nums[i] + nums[left] + nums[right]).into();
                if sum == target {
                    return target as i32;
                } else if sum > target {
                    right -= 1;
                } else if sum < target {
                    left += 1;
                }
                if (target - sum).abs() < (target - res).abs() {
                    res = sum;
                }
            }
        }
        res as i32
    }
}

// Letter Combinations of a Phone Number
impl Solution {
    pub fn letter_combinations(digits: String) -> Vec<String> {
        let mut res: Vec<String> = vec![];

        if digits.len() == 0 {
            return res;
        }

        let m: std::collections::HashMap<char, &str> = [
            ('1', ""),
            ('2', "abc"),
            ('3', "def"),
            ('4', "ghi"),
            ('5', "jkl"),
            ('6', "mno"),
            ('7', "pqrs"),
            ('8', "tuv"),
            ('9', "wxyz"),
        ]
        .iter()
        .cloned()
        .collect();

        for c in digits.chars() {
            let letters = m.get(&c).unwrap();
            let mut tmp: Vec<String> = vec![];

            for cc in letters.chars() {
                if res.len() == 0 {
                    tmp.push(cc.to_string());
                } else {
                    for r in res.iter() {
                        tmp.push(r.to_owned() + &cc.to_string());
                    }
                }
            }

            res = tmp;
        }
        res
    }
}

// ZigZag Conversion
impl Solution {
    pub fn convert(s: String, num_rows: i32) -> String {
        let mut buffer: Vec<Vec<char>> = vec![];
        for i in 0..num_rows {
            buffer.push(vec![]);
        }

        let chars: Vec<char> = s.chars().collect();
        let mut column = 0usize;
        while column < s.len() {
            let mut row = 0usize;
            while row < num_rows as usize && column < s.len() {
                buffer[row].push(chars[column]);
                row += 1;
                column += 1;
            }

            row = (num_rows as usize).checked_sub(2).unwrap_or(0);
            while row >= 1 && column < s.len() {
                buffer[row].push(chars[column]);
                row -= 1;
                column += 1;
            }
        }

        let mut res = "".to_string();
        for i in 0..num_rows {
            res = res + &(&buffer[i as usize]).iter().collect::<String>();
        }
        res
    }
}

// Palindrome Number
impl Solution {
    pub fn is_palindrome(x: i32) -> bool {
        if x < 0 || (x % 10 == 0 && x != 0) {
            return false;
        }

        let (mut x, mut rev) = (x, 0);
        while x > rev {
            rev = rev * 10 + x % 10;
            x /= 10;
        }
        x == rev || x == rev / 10
    }
}

// 4Sum
impl Solution {
    pub fn four_sum(nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut res: Vec<Vec<i32>> = vec![];
        if nums.len() < 4 {
            return res;
        }
        let mut nums = nums;
        nums.sort();

        for i in 0..nums.len() - 1 {
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }

            for j in i + 1..nums.len() - 2 {
                if j > i + 1 && nums[j] == nums[j - 1] {
                    continue;
                }

                let (mut left, mut right) = (j + 1, nums.len() - 1);
                while left < right {
                    let tmp = nums[i] + nums[j] + nums[left] + nums[right];
                    if tmp == target {
                        res.push([nums[i], nums[j], nums[left], nums[right]].to_vec());
                        left += 1;
                        right -= 1;

                        while left < right && nums[left] == nums[left - 1] {
                            left += 1;
                        }
                        while left < right && nums[right] == nums[right + 1] {
                            right -= 1;
                        }
                    } else if tmp < target {
                        left += 1;
                    } else if tmp > target {
                        right -= 1;
                    }
                }
            }
        }
        res
    }
}

// Regular Expression Matching
impl Solution {
    pub fn is_match(s: String, p: String) -> bool {
        fn is_match_str(s: &str, p: &str) -> bool {
            let (s_len, p_len) = (s.len(), p.len());
            if p_len == 0 {
                return s_len == 0;
            }

            let m = { s_len > 0 && (s.as_bytes()[0] == p.as_bytes()[0] || p.as_bytes()[0] == 46) };

            if p_len >= 2 && p.as_bytes()[1] == 42 {
                return is_match_str(s, &p[2..]) || (m && is_match_str(&s[1..], p));
            }

            m && is_match_str(&s[1..], &p[1..])
        }
        is_match_str(&s, &p)
    }
}

// Container With Most Water
impl Solution {
    pub fn max_area(height: Vec<i32>) -> i32 {
        let (mut res, mut left, mut right) = (0, 0, height.len() - 1);

        while left < right {
            let x = (right - left) as i32;
            let mut y = 0;

            if height[left] < height[right] {
                y = height[left];
                left += 1;
            } else {
                y = height[right];
                right -= 1;
            }

            res = res.max(x * y);
        }
        res
    }
}

// Definition for singly-linked list.
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct ListNode {
//   pub val: i32,
//   pub next: Option<Box<ListNode>>
// }
//
// impl ListNode {
//   #[inline]
//   fn new(val: i32) -> Self {
//     ListNode {
//       next: None,
//       val
//     }
//   }
// }
// Remove Nth Node From End of List
impl Solution {
    pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
        let mut dummy = Box::new(ListNode {
            val: -1,
            next: head,
        });

        let mut fast = dummy.clone();
        for _ in 0..n {
            fast = fast.next.unwrap();
        }

        let mut slow = dummy.as_mut();
        while let Some(n) = fast.next {
            fast = n;
            slow = slow.next.as_mut().unwrap();
        }

        slow.next = slow.next.as_mut().unwrap().next.clone();
        dummy.next
    }
}

// Definition for singly-linked list.
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct ListNode {
//   pub val: i32,
//   pub next: Option<Box<ListNode>>
// }
//
// impl ListNode {
//   #[inline]
//   fn new(val: i32) -> Self {
//     ListNode {
//       next: None,
//       val
//     }
//   }
// }
// Merge k Sorted Lists
impl Solution {
    pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
        use std::collections::BinaryHeap;
        fn build(
            mut head: Option<Box<ListNode>>,
            mut heap: BinaryHeap<i32>,
        ) -> Option<Box<ListNode>> {
            while !heap.is_empty() {
                head = Some(Box::new(ListNode {
                    val: heap.pop().unwrap(),
                    next: head,
                }));
            }
            head
        }
        build(
            None,
            BinaryHeap::from(lists.iter().fold(vec![], |mut acc, mut cur| {
                while cur.is_some() {
                    acc.push(cur.as_ref().unwrap().val);
                    cur = &cur.as_ref().unwrap().next;
                }
                acc
            })),
        )
    }
}

// Definition for singly-linked list.
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct ListNode {
//   pub val: i32,
//   pub next: Option<Box<ListNode>>
// }
//
// impl ListNode {
//   #[inline]
//   fn new(val: i32) -> Self {
//     ListNode {
//       next: None,
//       val
//     }
//   }
// }
// Reverse Nodes in k-Group
impl Solution {
    pub fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        fn add(head: Option<Box<ListNode>>, tail: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
            let mut head = head;
            let mut tail = tail;

            while let Some(mut new_tail) = head.take() {
                head = new_tail.next.take();
                new_tail.next = tail.take();
                tail = Some(new_tail);
            }
            tail
        }
        let mut head = head;
        let mut tail = &mut head;
        for _ in 0..k {
            match tail.as_mut() {
                None => return head,
                Some(tail_ref) => tail = &mut tail_ref.next,
            }
        }
        let tail = tail.take();
        add(head, Solution::reverse_k_group(tail, k))
    }
}

// Implement strStr()
impl Solution {
    pub fn str_str(haystack: String, needle: String) -> i32 {
        if needle.len() < 1 {
            return 0;
        }

        if haystack.len() < needle.len() {
            return -1;
        }

        let hay: Vec<char> = haystack.chars().collect();
        let nee: Vec<char> = needle.chars().collect();

        for i in 0..hay.len() {
            if hay[i] != nee[0] {
                continue;
            }

            let mut first = i as i32;
            let mut j = i + 1;

            for k in 1..nee.len() {
                if j >= hay.len() {
                    return -1;
                }

                if hay[j] == nee[k] {
                    j += 1;
                } else {
                    first = -1;
                    break;
                }
            }
            if first != -1 {
                return first;
            }
        }
        -1
    }
}

// Next Permutation
impl Solution {
    pub fn next_permutation(nums: &mut Vec<i32>) {
        match (1..nums.len()).rev().find(|&i| nums[i - 1] < nums[i]) {
            Some(i) => {
                let j = (i..nums.len())
                    .rev()
                    .find(|&j| nums[i - 1] < nums[j])
                    .unwrap();
                nums.swap(i - 1, j);
                nums[i..].reverse();
            }
            None => nums.reverse(),
        }
    }
}

// Valid Sudoku
impl Solution {
    pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
        let mut rows = [[false; 9]; 9];
        let mut cols = [[false; 9]; 9];
        let mut blks = [[false; 9]; 9];

        for i in 0..9 {
            for j in 0..9 {
                if board[i][j] == '.' {
                    continue;
                }

                let v = board[i][j] as usize - '1' as usize;
                let p = (i / 3 * 3) + (j / 3);

                if rows[i][v] || cols[j][v] || blks[p][v] {
                    return false;
                }

                rows[i][v] = true;
                cols[j][v] = true;
                blks[p][v] = true;
            }
        }
        true
    }
}

// Jump Game II
impl Solution {
    pub fn jump(nums: Vec<i32>) -> i32 {
        if nums.len() == 0 {
            return 1;
        }

        let (mut steps, mut current, mut end) = (0, 0, 0);
        for i in 0..nums.len() - 1 {
            let j = i as i32 + nums[i];
            if current < j {
                current = j;
            }
            if i == end {
                steps += 1;
                end = current as usize;
            }
        }
        steps
    }
}

// Rotate Image
impl Solution {
    pub fn rotate(matrix: &mut Vec<Vec<i32>>) {
        matrix.reverse();
        for i in 0..matrix.len() {
            for j in i + 1..matrix.len() {
                matrix[i][j] ^= matrix[j][i];
                matrix[j][i] = matrix[i][j] ^ matrix[j][i];
                matrix[i][j] ^= matrix[j][i];
            }
        }
    }
}

// Combination Sum
impl Solution {
    pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        fn backtrack(sub: &[i32], candidates: &[i32], target: i32, res: &mut Vec<Vec<i32>>) {
            let sum = sub.iter().sum::<i32>();
            if sum == target {
                res.push(sub.to_vec());
                return;
            } else if sum > target {
                return;
            }

            for (i, v) in candidates.iter().enumerate() {
                let mut s = sub.to_vec();
                s.push(*v);
                backtrack(&s, &candidates[i..], target, res);
            }
        }

        let mut res: Vec<Vec<i32>> = vec![];
        backtrack(&vec![], &candidates, target, &mut res);
        res
    }
}

// Combination Sum II
impl Solution {
    pub fn combination_sum2(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        fn backtrack(sub: &[i32], candidates: &[i32], target: i32, res: &mut Vec<Vec<i32>>) {
            let sum = sub.iter().sum::<i32>();
            if sum == target {
                res.push(sub.to_vec());
                return;
            } else if sum > target {
                return;
            }

            for (i, v) in candidates.iter().enumerate() {
                if i > 0 && candidates[i - 1] == *v {
                    continue;
                }
                let mut s = sub.to_vec();
                s.push(*v);
                backtrack(&s, &candidates[i + 1..], target, res);
            }
        }
        let mut candidates = candidates;
        candidates.sort();
        let mut res: Vec<Vec<i32>> = vec![];
        backtrack(&vec![], &candidates, target, &mut res);
        res
    }
}

// Trapping Rain Water
impl Solution {
    pub fn trap(height: Vec<i32>) -> i32 {
        let (mut l, mut r) = (0, height.len() - 1);
        let (mut lm, mut rm) = (0, 0);
        let mut res = 0;

        while l < r && r != std::usize::MAX {
            if height[l] < height[r] {
                if height[l] >= lm {
                    lm = height[l];
                } else {
                    res += lm - height[l];
                }
                l += 1;
            } else {
                if height[r] >= rm {
                    rm = height[r];
                } else {
                    res += rm - height[r];
                }
                r -= 1;
            }
        }
        res
    }
}

// Group Anagrams
impl Solution {
    pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
        use std::collections::HashMap;

        let mut res: Vec<Vec<String>> = vec![];
        let mut m: HashMap<[u8; 26], usize> = HashMap::new();
        let mut i = 0;

        for str in strs.iter() {
            let mut s: [u8; 26] = [0; 26];
            for c in str.chars() {
                let ci = c as usize - 'a' as usize;
                s[ci] += 1;
            }
            match m.get(&s) {
                Some(j) => {
                    res[*j].push(str.to_string());
                }
                None => {
                    m.insert(s, i);
                    if res.len() < i + 1 {
                        res.push(vec![]);
                    }
                    res[i].push(str.to_string());
                    i += 1;
                }
            }
        }
        res
    }
}
// amazing example solution
// https://leetcode.com/problems/group-anagrams/discuss/566237/Rust-Solution
//
// impl Solution {
//     pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
//         let mut out = std::collections::HashMap::new();
//         for v in strs {
//             let mut k: Vec<u8> = v.bytes().collect();
//             k.sort_unstable();
//             out.entry(k).or_insert_with(|| vec![]).push(v)
//         }
//         out.into_iter().map(|(_, v)| v).collect()
//     }
// }
//

// Permutations
impl Solution {
    pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
        fn backtrack(nums: &[i32], sub: &[i32], res: &mut Vec<Vec<i32>>) {
            if nums.len() == 0 {
                res.push(sub.to_vec());
                return;
            }
            for (i, v) in nums.iter().enumerate() {
                let (mut nums_c, mut sub_c) = (nums.to_vec(), sub.to_vec());
                nums_c.remove(i as usize);
                sub_c.push(*v);
                backtrack(&nums_c, &sub_c, res);
            }
        }
        let mut res: Vec<Vec<i32>> = vec![];
        backtrack(&nums, &vec![], &mut res);
        res
    }
}

// Multiply Strings
impl Solution {
    pub fn multiply(num1: String, num2: String) -> String {
        let mut res: Vec<u8> = vec![0; num1.len() + num2.len()];
        let n1: Vec<char> = num1.chars().collect();
        let n2: Vec<char> = num2.chars().collect();
        let mut i = num2.len() - 1;

        while i != std::usize::MAX {
            let mut j = num1.len() - 1;
            while j != std::usize::MAX {
                let (v1, v2) = (n1[j] as u8 - '0' as u8, n2[i] as u8 - '0' as u8);
                let v = (v1 * v2) + res[i + j + 1];
                res[i + j] = v / 10 + res[i + j];
                res[i + j + 1] = v - (v / 10) * 10;
                j -= 1;
            }
            i -= 1;
        }

        let mut idx = num1.len() + num2.len() - 1;
        let mut i = idx;
        while i != std::usize::MAX {
            if res[i as usize] > 0 {
                idx = i;
            }
            res[i as usize] += '0' as u8;
            i -= 1;
        }

        std::str::from_utf8(&res[idx..]).unwrap().to_string()
    }
}

// Jump Game
impl Solution {
    pub fn can_jump(nums: Vec<i32>) -> bool {
        nums.iter().enumerate().fold(0, |acc, (i, v)| {
            if acc < i as i32 {
                return -1;
            }
            acc.max(i as i32 + *v)
        }) >= (nums.len() - 1) as i32
    }
}

// Length of Last Word
impl Solution {
    pub fn length_of_last_word(s: String) -> i32 {
        let mut count = 0;
        let mut re_count = false;

        for c in s.chars() {
            if c == ' ' {
                re_count = true;
            } else {
                if re_count {
                    count = 0;
                    re_count = false;
                }
                count += 1;
            }
        }
        count
    }
}

// Unique Binary Search Trees
impl Solution {
    pub fn num_trees(n: i32) -> i32 {
        let n = n as usize;
        let mut dp: Vec<i32> = vec![0; n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for i in 2..=n {
            for j in 1..=i {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        dp[n]
    }
}

// Plus One
impl Solution {
    pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
        let mut digits = digits;
        for v in digits.iter_mut().rev() {
            let sum = *v + 1;
            *v = sum % 10;
            if sum < 10 {
                return digits;
            }
        }
        [&vec![1], &digits[..]].concat()
    }
}

// Unique Paths
impl Solution {
    pub fn unique_paths(m: i32, n: i32) -> i32 {
        if m == 0 || n == 0 {
            return 0;
        }

        let m: usize = m as usize;
        let n: usize = n as usize;

        let mut path: Vec<Vec<i32>> = vec![vec![0; n]; m];
        for i in 0..m {
            for j in 0..n {
                if i == 0 || j == 0 {
                    path[i][j] = 1;
                } else {
                    path[i][j] = path[i - 1][j] + path[i][j - 1];
                }
            }
        }
        path[m - 1][n - 1]
    }
}

// Unique Paths II
impl Solution {
    pub fn unique_paths_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
        let row_len = obstacle_grid.len();
        let col_len = obstacle_grid[0].len();
        let mut dp = vec![vec![0; col_len]; row_len];
        for i in 0..row_len {
            for j in 0..col_len {
                if obstacle_grid[i][j] == 1 {
                    dp[i][j] = 0;
                    continue;
                }
                if i == 0 && j == 0 {
                    dp[i][j] = 1;
                } else if i == 0 {
                    dp[i][j] = dp[i][j - 1];
                } else if j == 0 {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        dp[row_len - 1][col_len - 1]
    }
}

// Climbing Stairs
impl Solution {
    pub fn climb_stairs(n: i32) -> i32 {
        if n < 3 {
            return n;
        }
        let (mut p1, mut p2, mut res) = (1, 2, 0);
        for i in 2..n {
            res = p1 + p2;
            p1 = p2;
            p2 = res;
        }
        res
    }
}
