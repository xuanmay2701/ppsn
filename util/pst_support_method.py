def find_c_shapelet(list_ppi, c):
    sort_list_ppi = list_ppi[list_ppi[:, 3].argsort()]
    return sort_list_ppi[-c:]


def find_c_shapelet_non_overlab(list_ppi, c, p=0.95):
    sort_list_ppi = list_ppi[list_ppi[:, 3].argsort()]
    list_group_shapelet = [sort_list_ppi[-1]]
    list_pos = [len(sort_list_ppi)-1]
    for i in reversed(range(len(sort_list_ppi) - 1)):
        is_add = True
        s2 = int(sort_list_ppi[i, 1])
        e2 = int(sort_list_ppi[i, 2])
        for j in range(len(list_group_shapelet)):
            s1 = int(list_group_shapelet[j][1])
            e1 = int(list_group_shapelet[j][2])
            if (s1 <= s2 and e1 >= e2) or (s2 <= s1 and e2 >= e1):
                is_add = False
                break
            elif s1 < s2 and e1 < e2:
                if (e1 - s2) / min((e2 - s2), (e1 - s1)) > p:
                    is_add = False
                    break
            elif s1 > s2 and e1 > e2:
                if (e2 - s1) / min((e2 - s2), (e1 - s1)) > p:
                    is_add = False
                    break
        if is_add:
            list_group_shapelet.append(sort_list_ppi[i])
            list_pos.append(i)
            if len(list_group_shapelet) == c:
                break
    if len(list_group_shapelet) < c:
        for i in reversed(range(len(sort_list_ppi) - 1)):
            if i not in list_pos:
                list_group_shapelet.append(sort_list_ppi[i])
                if len(list_group_shapelet) == c:
                    break
    return list_group_shapelet