import stuett


def get_images_from_timestamps(store, start, end):
    return stuett.data.MHDSLRFilenames(store=store,
                                       start_time=start,
                                       end_time=end,
                                       as_pandas=True)

